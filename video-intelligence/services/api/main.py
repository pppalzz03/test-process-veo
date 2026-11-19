"""FastAPI app: create/inspect video jobs and read results. Thin layer over use cases."""

import hmac
import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)

from services.api import service
from shared.config.settings import Settings, get_settings
from shared.database import jobs as jobs_db
from shared.database import results as results_db
from shared.database.pool import DictConnection, DictPool, create_pool
from shared.database.videos import get_video
from shared.logging import configure_logging
from shared.schemas.api import (
    DetectionOut,
    EventOut,
    JobCreateIn,
    JobOut,
    Page,
    RegisterVideoIn,
    SegmentOut,
    TrackOut,
    VideoOut,
)

logger = logging.getLogger("api")


def create_app(settings: Settings | None = None, pool: DictPool | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        configure_logging(cfg.log_level, json_output=cfg.log_json, service="api")
        app.state.pool = pool or create_pool(cfg.database_url)
        yield
        if pool is None:
            app.state.pool.close()

    app = FastAPI(title="Video Intelligence API", version="0.3.0", lifespan=lifespan)
    app.state.settings = cfg

    def get_conn(request: Request) -> Iterator[DictConnection]:
        with request.app.state.pool.connection() as conn:
            yield conn

    def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
        if cfg.api_key and not (x_api_key and hmac.compare_digest(x_api_key, cfg.api_key)):
            raise HTTPException(401, "invalid or missing X-API-Key")

    Conn = Annotated[DictConnection, Depends(get_conn)]
    Auth = [Depends(require_api_key)]

    # ------------------------------------------------------------------ health
    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready(conn: Conn) -> dict[str, str]:
        try:
            conn.execute("SELECT 1")
        except Exception as exc:
            raise HTTPException(503, "database unavailable") from exc
        return {"status": "ok"}

    # ------------------------------------------------------------------ videos
    @app.post("/v1/videos", response_model=VideoOut, status_code=201, dependencies=Auth)
    def upload_video(conn: Conn, response: Response, file: Annotated[UploadFile, File()]):  # type: ignore[no-untyped-def]
        try:
            video, created = service.store_uploaded_video(conn, cfg, file.file)
        except service.VideoRejected as exc:
            raise HTTPException(422, str(exc)) from exc
        if not created:
            response.status_code = 200  # same content already stored
        return video

    @app.post("/v1/videos/register", response_model=VideoOut, status_code=201, dependencies=Auth)
    def register_video(conn: Conn, response: Response, body: RegisterVideoIn):  # type: ignore[no-untyped-def]
        try:
            video, created = service.register_existing_video(conn, cfg, body.path)
        except service.VideoRejected as exc:
            raise HTTPException(422, str(exc)) from exc
        if not created:
            response.status_code = 200
        return video

    @app.get("/v1/videos/{video_id}", response_model=VideoOut, dependencies=Auth)
    def read_video(video_id: UUID, conn: Conn):  # type: ignore[no-untyped-def]
        video = get_video(conn, video_id)
        if video is None:
            raise HTTPException(404, "video not found")
        return video

    # -------------------------------------------------------------------- jobs
    @app.post("/v1/jobs", response_model=JobOut, status_code=202, dependencies=Auth)
    def create_job(conn: Conn, response: Response, body: JobCreateIn):  # type: ignore[no-untyped-def]
        try:
            job, created = service.create_processing_job(
                conn, cfg, body.video_id, body.idempotency_key
            )
        except service.VideoRejected as exc:
            raise HTTPException(404, str(exc)) from exc
        if not created:
            response.status_code = 200  # idempotent replay
        return job

    @app.get("/v1/jobs", response_model=list[JobOut], dependencies=Auth)
    def list_jobs(  # type: ignore[no-untyped-def]
        conn: Conn,
        status: str | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        return jobs_db.list_jobs(conn, status=status, limit=limit, offset=offset)

    def _job_or_404(conn: DictConnection, job_id: UUID) -> dict:
        job = jobs_db.get_job(conn, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job

    @app.get("/v1/jobs/{job_id}", response_model=JobOut, dependencies=Auth)
    def read_job(job_id: UUID, conn: Conn):  # type: ignore[no-untyped-def]
        return _job_or_404(conn, job_id)

    @app.get("/v1/jobs/{job_id}/segments", response_model=list[SegmentOut], dependencies=Auth)
    def read_segments(job_id: UUID, conn: Conn):  # type: ignore[no-untyped-def]
        _job_or_404(conn, job_id)
        return results_db.list_segments(conn, job_id)

    @app.get("/v1/jobs/{job_id}/tracks", response_model=Page[TrackOut], dependencies=Auth)
    def read_tracks(  # type: ignore[no-untyped-def]
        job_id: UUID,
        conn: Conn,
        class_name: str | None = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
        cursor: Annotated[int, Query(ge=0)] = 0,
    ):
        _job_or_404(conn, job_id)
        rows = results_db.list_tracks(
            conn, job_id, class_name=class_name, limit=limit, after_id=cursor
        )
        return Page(items=rows, next_cursor=rows[-1]["id"] if len(rows) == limit else None)

    @app.get("/v1/jobs/{job_id}/events", response_model=list[EventOut], dependencies=Auth)
    def read_events(  # type: ignore[no-untyped-def]
        job_id: UUID,
        conn: Conn,
        event_type: str | None = None,
        class_name: str | None = None,
        from_ms: Annotated[int | None, Query(ge=0)] = None,
        to_ms: Annotated[int | None, Query(ge=0)] = None,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        _job_or_404(conn, job_id)
        return results_db.list_events(
            conn,
            job_id,
            event_type=event_type,
            class_name=class_name,
            from_ms=from_ms,
            to_ms=to_ms,
            limit=limit,
            offset=offset,
        )

    @app.get("/v1/jobs/{job_id}/detections", response_model=Page[DetectionOut], dependencies=Auth)
    def read_detections(  # type: ignore[no-untyped-def]
        job_id: UUID,
        conn: Conn,
        track_id: int | None = None,
        from_ms: Annotated[int | None, Query(ge=0)] = None,
        to_ms: Annotated[int | None, Query(ge=0)] = None,
        limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
        cursor: Annotated[int, Query(ge=0)] = 0,
    ):
        _job_or_404(conn, job_id)
        rows = results_db.list_detections(
            conn,
            job_id,
            track_id=track_id,
            from_ms=from_ms,
            to_ms=to_ms,
            limit=limit,
            after_id=cursor,
        )
        return Page(items=rows, next_cursor=rows[-1]["id"] if len(rows) == limit else None)

    return app


def app_factory() -> FastAPI:
    """uvicorn --factory services.api.main:app_factory"""
    return create_app()
