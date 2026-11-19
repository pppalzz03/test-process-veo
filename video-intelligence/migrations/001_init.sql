-- Phase 1-3 schema. PostgreSQL is the source of truth.
-- Phase 4 (VLM / Qdrant) adds new tables and outbox routing keys; nothing here needs to change:
--   * events.revision + events.evidence  -> VLM captions / embeddings reference (event_id, revision)
--   * outbox.exchange/routing_key        -> new queues (video.vlm, video.embed) need no schema change

CREATE TABLE videos (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id       text        NOT NULL DEFAULT 'default',
    storage_key     text        NOT NULL,          -- path relative to MEDIA_ROOT
    sha256          text        NOT NULL,
    size_bytes      bigint      NOT NULL,
    duration_ms     bigint      NOT NULL CHECK (duration_ms > 0),
    start_offset_ms bigint      NOT NULL DEFAULT 0, -- container start_time; subtracted from PTS
    fps             double precision,
    width           integer,
    height          integer,
    codec           text,
    status          text        NOT NULL DEFAULT 'ready',
    created_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, sha256)
);

CREATE TABLE jobs (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    video_id         uuid        NOT NULL REFERENCES videos(id),
    status           text        NOT NULL
                     CHECK (status IN ('queued','running','retry_wait','completed','failed')),
    idempotency_key  text UNIQUE,
    model_version    text        NOT NULL,
    config           jsonb       NOT NULL,           -- ProcessingConfig snapshot
    config_hash      text        NOT NULL,
    attempts         integer     NOT NULL DEFAULT 0,
    max_attempts     integer     NOT NULL,
    next_attempt_at  timestamptz,
    worker_id        text,
    lease_expires_at timestamptz,
    error_code       text,
    error_message    text,
    created_at       timestamptz NOT NULL DEFAULT now(),
    started_at       timestamptz,                    -- first time a worker picked it up
    completed_at     timestamptz,
    updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX jobs_status_idx ON jobs (status, created_at);
CREATE INDEX jobs_lease_idx  ON jobs (lease_expires_at) WHERE status = 'running';
CREATE INDEX jobs_video_idx  ON jobs (video_id);

CREATE TABLE segments (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id           uuid        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    segment_index    integer     NOT NULL,
    start_ms         bigint      NOT NULL,
    end_ms           bigint      NOT NULL,
    status           text        NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','running','done')),
    frames_processed integer,
    detections_count integer,
    processing_ms    bigint,
    completed_at     timestamptz,
    UNIQUE (job_id, segment_index),
    CHECK (end_ms > start_ms)
);

-- Track IDs are SEGMENT-LOCAL (DeepStream tracker restarts per pipeline run).
CREATE TABLE tracks (
    id              bigserial PRIMARY KEY,
    segment_id      uuid        NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    local_track_id  bigint      NOT NULL,
    class_id        integer     NOT NULL,
    class_name      text        NOT NULL,
    first_seen_ms   bigint      NOT NULL,
    last_seen_ms    bigint      NOT NULL,
    frame_count     integer     NOT NULL,
    avg_confidence  real        NOT NULL,
    UNIQUE (segment_id, local_track_id)
);

-- High volume. Retention/partitioning is a documented follow-up (docs/architecture.md).
CREATE TABLE detections (
    id            bigserial PRIMARY KEY,
    segment_id    uuid     NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    track_id      bigint   REFERENCES tracks(id) ON DELETE CASCADE,
    frame_index   integer  NOT NULL,
    timestamp_ms  bigint   NOT NULL,
    class_id      integer  NOT NULL,
    class_name    text     NOT NULL,
    confidence    real     NOT NULL,
    bbox_left     real     NOT NULL,
    bbox_top      real     NOT NULL,
    bbox_width    real     NOT NULL,
    bbox_height   real     NOT NULL
);
CREATE INDEX detections_segment_time_idx ON detections (segment_id, timestamp_ms);
CREATE INDEX detections_track_idx        ON detections (track_id);

CREATE TABLE events (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id       uuid        NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    segment_id   uuid        NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    event_key    text        NOT NULL,
    event_type   text        NOT NULL,
    start_ms     bigint      NOT NULL,
    end_ms       bigint      NOT NULL,
    class_name   text,
    description  text,
    attributes   jsonb       NOT NULL DEFAULT '{}'::jsonb,
    evidence     jsonb,                     -- reserved for Phase 4 (keyframe timestamps / URIs)
    revision     integer     NOT NULL DEFAULT 1,
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (job_id, event_key),             -- duplicate-event guard across retries
    CHECK (end_ms >= start_ms)
);
CREATE INDEX events_job_time_idx ON events (job_id, start_ms);
CREATE INDEX events_type_idx     ON events (event_type, class_name);

CREATE TABLE event_objects (
    event_id  uuid   NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    track_id  bigint NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
    PRIMARY KEY (event_id, track_id)
);

-- Transactional outbox: written in the same transaction as the state change it announces.
CREATE TABLE outbox (
    id               bigserial PRIMARY KEY,
    aggregate_id     uuid        NOT NULL,
    event_type       text        NOT NULL,
    exchange         text        NOT NULL DEFAULT 'video',
    routing_key      text        NOT NULL,
    payload          jsonb       NOT NULL,
    available_at     timestamptz NOT NULL DEFAULT now(),  -- future time => delayed retry
    published_at     timestamptz,
    publish_attempts integer     NOT NULL DEFAULT 0,
    last_error       text,
    created_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX outbox_unpublished_idx ON outbox (available_at, id) WHERE published_at IS NULL;

-- Single-GPU admission: N slots, each held by one running segment with a renewable lease.
CREATE TABLE gpu_slots (
    slot_id          integer PRIMARY KEY,
    holder           text,
    acquired_at      timestamptz,
    lease_expires_at timestamptz
);

CREATE TABLE service_heartbeats (
    service    text PRIMARY KEY,
    beat_at    timestamptz NOT NULL DEFAULT now()
);
