# Deployment and validation on the DGX Spark

Run everything below **on the DGX Spark**. Nothing here stops, restarts or reconfigures vLLM or
any existing container. There is deliberately no `down -v` / `prune` step.

## 0. Prerequisites

* Docker + Compose + NVIDIA Container Toolkit (already installed per the host baseline).
* Network access to pull `postgres:16`, `rabbitmq:4-management` and the DeepStream image.
* A YOLO weights file and at least one real MP4 (ideally one of >= 1 hour, H.264/H.265).

## 1. Phase 1 - validate the environment (read-only)

```bash
scripts/env_check.sh > /tmp/env-host.txt
scripts/env_check.sh --container-test --pull > /tmp/env-container.txt   # pulls the DeepStream image
```

Paste both into `docs/environment-report.md` and tick its checklist. If the image fails the GPU
or `pyds` checks, read "Open compatibility risks" there (fallback: 9.1 image via `.env`).

## 2. Configure

```bash
cp .env.example .env
# edit .env: set URL-safe POSTGRES_PASSWORD / RABBITMQ_PASSWORD, MODEL_VERSION, optionally API_KEY
mkdir -p data/media/uploads && sudo chown 10001 data/media/uploads   # API container user
cp /path/to/videos/*.mp4 data/media/
```

## 3. Model: export, inspect, build the engine once

```bash
pip install ultralytics onnx
python scripts/export_yolo.py --weights yolo11n.pt --out deepstream/models/yolo.onnx
python scripts/inspect_yolo_model.py deepstream/models/yolo.onnx
```

* Output `[4+nc, N]` -> keep `DS_INFER_CONFIG=.../config_infer_primary_yolo.txt`.
* Output `[300, 6]` -> use `config_infer_primary_yolo_e2e.txt` (and name the file `yolo_e2e.onnx`).
* If the class count is not 80, edit `num-detected-classes` and `deepstream/models/labels.txt`.

```bash
docker compose build worker                   # also compiles the custom parser
make build-engine VIDEO=/data/media/sample.mp4    # builds + caches the FP16 TensorRT engine
```

## 4. Phase 2 - validate the pipeline on one real video (before the queue)

```bash
docker compose run --rm --no-deps worker bash -c '
  /opt/venv/bin/python /opt/vi/deepstream/app/ds_runner.py \
     --source /data/media/sample.mp4 --start-ms 0 --end-ms 10000 --codec h264 \
     --infer-config $DS_INFER_CONFIG \
     --tracker-config /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml \
     --tracker-lib /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so \
     --labels /opt/vi/models/labels.txt | head -3'
docker compose run --rm --no-deps worker /opt/venv/bin/python /app/scripts/validate_timestamps.py \
     /data/media/sample.mp4 --start-ms 60000 --end-ms 70000
```

Do not continue until: frames carry plausible detections, `validate_timestamps.py` prints
`TIMESTAMPS OK`, and `nvidia-smi dmon -s u` shows decoder activity (hardware decode).

## 5. Phase 3 - start the system

```bash
make up                  # postgres, rabbitmq, migrate, api, scheduler, worker
make ps                  # api / scheduler should become healthy
curl -s localhost:8000/health/ready

# upload (or POST /v1/videos/register {"path": "sample.mp4"} for a file already in data/media)
VID=$(curl -s -F file=@sample.mp4 localhost:8000/v1/videos | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
JOB=$(curl -s -X POST localhost:8000/v1/jobs -H 'Content-Type: application/json' \
      -d "{\"video_id\":\"$VID\",\"idempotency_key\":\"first-run\"}" | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
curl -s localhost:8000/v1/jobs/$JOB             # status, segments_done/segments_total, attempts
curl -s "localhost:8000/v1/jobs/$JOB/events?event_type=object_presence&class_name=person"
```

If `API_KEY` is set add `-H "X-API-Key: ..."`. Interactive API docs: `http://localhost:8000/docs`.
RabbitMQ UI: `http://localhost:15672` (localhost only).

Scale carefully: `docker compose up -d --scale worker=2` is allowed, but **real GPU concurrency
is capped by `GPU_MAX_CONCURRENT_STREAMS`** (default 1). Raise it only after the benchmark
shows memory headroom next to vLLM.

## 6. Tests

```bash
pip install -e '.[dev]'
make lint typecheck test parser-test
RABBITMQ_URL=amqp://vi:<password>@localhost:5672/%2F make test-rabbitmq   # disposable broker only
```

Recovery tests to run by hand on the Spark (record results in README status):

1. **Worker crash:** submit a 1 h job, `docker kill <worker>` mid-way, `docker compose up -d worker`.
   Expect: job completes after lease expiry (<= `LEASE_SECONDS` + reaper interval), finished
   segments are not redone (check logs), `SELECT event_key, count(*) FROM events GROUP BY 1 HAVING count(*)>1` returns 0 rows.
2. **Broker restart:** `docker compose restart rabbitmq` while jobs queue; no job lost.
3. **Corrupt video:** `head -c 5000000 good.mp4 > bad.mp4`; expect `failed` with
   `bad_input` or `low_frame_coverage`, and a message in `video.dlq`.
4. **Duplicate delivery:** re-publish a job message from the RabbitMQ UI; expect no new events.
5. **vLLM untouched:** compare `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
   before/during/after - vLLM PIDs and memory unchanged.

## 7. Benchmarks (real data only)

Run in this order; vary only one thing at a time (see `docs/benchmark-report.md` checklist):

```bash
python scripts/benchmark_yolo_baseline.py --weights yolo11n.pt --video data/media/sample.mp4 --half
docker compose run --rm --no-deps worker /opt/venv/bin/python /app/scripts/benchmark_pipeline.py \
     --video /data/media/sample.mp4 --streams 1 --warmup      # then --streams 2, then 4
python scripts/benchmark_batch.py --video sample.mp4 --video long_1h.mp4 --jobs-per-video 3
python scripts/render_benchmark_report.py                      # rewrites docs/benchmark-report.md
```

Results are written to `./benchmarks/results/` on the host (bind-mounted into the worker).
Check `all_streams_reached_eos` and frames-vs-expected before reading any FPS number.

## 8. Operations

| Task | Command |
|---|---|
| Logs (JSON) | `make logs` |
| Stuck/failed jobs | `curl localhost:8000/v1/jobs?status=failed` |
| Dead letters | RabbitMQ UI -> queue `video.dlq` |
| Re-run a failed job | create a new job with a new `idempotency_key` |
| Stop (keep data) | `make down` |
| Roll back this deployment | `docker compose down` (volumes `pgdata`, `rabbitmq-data` remain; remove them only after a verified `pg_dump` backup and explicit approval) |

Backup: `docker compose exec postgres pg_dump -U vi vi > backup.sql` (test the restore).
