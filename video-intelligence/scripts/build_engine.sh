#!/usr/bin/env bash
# Build (and cache) the TensorRT engine ONCE, before starting workers, by running the real
# DeepStream pipeline on the first second of a video. nvinfer serializes the engine to the
# model-engine-file path in the nvinfer config. Several workers must not build it concurrently.
#
#   docker compose run --rm --no-deps worker scripts/build_engine.sh /data/media/sample.mp4
#
# Engine files are tied to the GPU + TensorRT version: build on the DGX Spark, never copy them.
set -euo pipefail
VIDEO="${1:?usage: build_engine.sh /data/media/<video.mp4> [codec]}"
CODEC="${2:-}"
RUNNER=(${RUNNER_COMMAND:-/opt/venv/bin/python /opt/vi/deepstream/app/ds_runner.py})
ARGS=(--source "$VIDEO" --start-ms 0 --end-ms 1000
      --infer-config "${DS_INFER_CONFIG:-/opt/vi/deepstream/configs/config_infer_primary_yolo.txt}"
      --tracker-config "${DS_TRACKER_CONFIG:-/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml}"
      --tracker-lib "${DS_TRACKER_LIB:-/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so}"
      --labels "${DS_LABELS_FILE:-/opt/vi/models/labels.txt}")
[ -n "$CODEC" ] && ARGS+=(--codec "$CODEC")
echo "building TensorRT engine (first run can take several minutes)..." >&2
time "${RUNNER[@]}" "${ARGS[@]}" | tail -2
ls -l /opt/vi/models/*.engine
