# Video Intelligence - batch video analytics on DGX Spark (Phase 1-3)

`Video Upload -> RabbitMQ -> GPU Scheduler -> DeepStream + YOLO (TensorRT) -> PostgreSQL -> Result API`

Processes MP4 videos of an hour or more on a **single shared GPU** (NVIDIA DGX Spark GB10, ARM64)
without touching the existing vLLM workloads. PostgreSQL is the source of truth; RabbitMQ carries
job IDs; a transactional outbox guarantees no job is lost between commit and publish. VLM,
Qdrant, embeddings and the LLM agent are **out of scope** (Phase 4); the schema leaves hooks
for them (see `docs/architecture.md`).

> **Not production-ready.** The orchestration is tested; the DeepStream/GPU path and the real
> RabbitMQ path are **not yet executed on a DGX Spark**, no benchmark has been run, and the
> end-to-end 1-hour run is pending. See the status table.

## Verification status (what was actually run)

| Area | Status | Evidence |
|---|---|---|
| Domain logic (backoff, segmentation, state machine, event extraction, probe parsing) | **Tested** | 74 unit tests (incl. consumer ack logic, benchmark tooling) |
| PostgreSQL schema, migrations, outbox, leases, retries, DLQ rows, reaper, idempotent segment commit, GPU slots | **Tested on real PostgreSQL 16** | `tests/integration/test_db_lifecycle.py`; mutation-checked (breaking the claim/commit guards makes 4 tests fail) |
| Worker flow with a **real subprocess**: success, resume, duplicate delivery, corrupt video, transient retry -> DLQ, no-EOS, truncated video, graceful shutdown, lost lease, GPU slot cap | **Tested** with a synthetic runner test double | `tests/integration/test_worker_flow.py` |
| API: upload/register, dedup, path traversal, idempotent job creation, result endpoints, pagination, API key | **Tested** with real ffprobe + real MP4 + real PostgreSQL; live `uvicorn` + migrate CLI smoke test | `tests/integration/test_api.py` |
| Consumer ack/reject/redelivery logic | **Tested with a fake channel** | `tests/unit/test_consumer_delivery.py` |
| Benchmark harness mechanics (frame counting, EOS flag, synthetic flagging, report renders only measured data) | **Tested** | `tests/unit/test_benchmark_tools.py` |
| YOLO decode logic (C++) | **Compiled and tested on host** | `make -C deepstream/parser test` |
| `docker compose config` | **Valid** | no daemon in the authoring sandbox, so nothing was built or started |
| **RabbitMQ publisher/consumer/topology against a real broker** | **NOT RUN** (tests written, skipped) | `tests/integration/test_rabbitmq.py` |
| **DeepStream pipeline (`ds_runner.py`), hardware decode, TensorRT engine, tracker, custom parser glue** | **NOT RUN** (no GPU/DGX Spark/model/video available) | needs `docs/deployment.md` sections 1-4 |
| **Docker images build** | **NOT RUN** | `docker compose build` |
| 1-hour video end to end, recovery tests on the Spark, vLLM-untouched check | **NOT RUN** | `docs/deployment.md` section 6 |
| **Benchmarks** | **NOT RUN, no numbers exist** | `docs/benchmark-report.md` |

Test totals in the authoring sandbox: **141 passed, 7 skipped** (the 7 are the RabbitMQ tests);
`ruff` and `mypy` clean.

## Layout

```
compose.yaml  .env.example  Makefile  pyproject.toml
services/
  api/              FastAPI: videos, jobs, results (+ ffprobe validation)
  worker/           RabbitMQ consumer, job handler (retry policy), processor, runner adapter
  scheduler/        outbox relay, lease reaper, GPU slot scheduler
  event_processor/  pure functions: detections -> tracks -> events
shared/             config (pydantic), database (psycopg), messaging (pika), schemas, domain, logging
deepstream/
  app/ds_runner.py  DeepStream pipeline for one window -> NDJSON (runs in the DeepStream container)
  configs/          nvinfer configs (raw YOLO head, NMS-free head)
  parser/           custom YOLO bbox parser (C++) + host unit test
  models/           labels.txt (put yolo.onnx here)
migrations/         plain SQL, applied by `python -m shared.database.migrate`
rabbitmq/           rabbitmq.conf (24 h consumer_timeout), enabled_plugins
docker/             app.Dockerfile (control plane), worker.Dockerfile (DeepStream SBSA base)
scripts/            env_check.sh, export_yolo.py, inspect_yolo_model.py, build_engine.sh,
                    validate_timestamps.py, benchmark_*.py, render_benchmark_report.py
tests/              unit/, integration/, fixtures/ (synthetic runner test double)
docs/               environment-report.md, architecture.md, deployment.md, benchmark-report.md
```

## Quick start

On the DGX Spark, follow `docs/deployment.md` in order (environment check -> model export ->
single-video pipeline validation -> stack). Short version once the model/engine are ready:

```bash
cp .env.example .env            # set passwords, MODEL_VERSION
make up                         # postgres, rabbitmq, migrate, api, scheduler, worker
curl -F file=@sample.mp4 localhost:8000/v1/videos                       # -> video id
curl -X POST localhost:8000/v1/jobs -H 'Content-Type: application/json' \
     -d '{"video_id":"<id>","idempotency_key":"run-1"}'                  # -> job id (202)
curl localhost:8000/v1/jobs/<job_id>                                     # progress
curl "localhost:8000/v1/jobs/<job_id>/events?event_type=object_presence" # results
```

Develop and test without a GPU (needs PostgreSQL binaries or `TEST_DATABASE_URL`, ffmpeg):

```bash
pip install -e '.[dev]'
make lint typecheck test parser-test
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/videos` | multipart upload; streamed to disk, probed, deduplicated by SHA-256 |
| POST | `/v1/videos/register` | register a file already under `MEDIA_ROOT` |
| POST | `/v1/jobs` | create job (202); `idempotency_key` makes retries safe |
| GET | `/v1/jobs`, `/v1/jobs/{id}` | status, attempts, segment progress, error code |
| GET | `/v1/jobs/{id}/segments` / `tracks` / `events` / `detections` | results (cursor pagination for tracks/detections; time/type/class filters for events) |
| GET | `/health/live`, `/health/ready` | probes |

## Key configuration (see `.env.example`, `shared/config/settings.py`)

`GPU_MAX_CONCURRENT_STREAMS` (default 1), `GPU_MIN_AVAILABLE_MEM_MB`, `SEGMENT_SECONDS` (300),
`MIN_CONFIDENCE`, `MAX_ATTEMPTS`, `RETRY_BASE_SECONDS`, `LEASE_SECONDS`, `DS_INFER_CONFIG`,
`DEEPSTREAM_IMAGE`, `PERSIST_DETECTIONS`, `DETECTION_STRIDE`.

## Safety rules baked in

No component stops, restarts, signals or reconfigures vLLM or other containers. Compose creates
only its own network and volumes, and there is no `down -v`/`prune` target. GPU admission is
limited to our own slots and system memory headroom. Secrets live only in `.env` (git-ignored).

## Rollback

`make down` (volumes are kept). Remove volumes only after `pg_dump` and explicit approval.

## Not in this release

Cross-segment track stitching, batched multi-source pipelines, Prometheus metrics, TLS,
multi-tenant isolation, detection retention/partitioning, VLM/Qdrant/LLM (Phase 4).
