#!/usr/bin/env bash
# Phase 1 environment validation. READ-ONLY: it never stops, restarts or reconfigures anything.
#
#   scripts/env_check.sh                       # host inspection only
#   scripts/env_check.sh --container-test      # also run short, throw-away checks inside the
#                                              # DeepStream image IF it is already pulled
#   scripts/env_check.sh --container-test --pull   # additionally allow `docker pull` of the image
#
# Run it ON THE DGX SPARK and paste the output into docs/environment-report.md.
set -u
IMAGE="${DEEPSTREAM_IMAGE:-ghcr.io/nvidia/deepstream:9.2-triton-sbsa-dgx-spark}"
CONTAINER_TEST=0; PULL=0
for arg in "$@"; do
  case "$arg" in
    --container-test) CONTAINER_TEST=1 ;;
    --pull) PULL=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

section() { printf '\n## %s\n' "$1"; }
run() { printf '$ %s\n' "$*"; "$@" 2>&1 || echo "(exit $?)"; }
have() { command -v "$1" >/dev/null 2>&1; }

echo "# Environment check $(date -u +%FT%TZ) on $(hostname)"

section "Host"
run uname -m
run cat /etc/os-release
run free -h
run df -h /
have nproc && run nproc

section "GPU and driver"
if have nvidia-smi; then
  run nvidia-smi
  run nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv
  echo; echo "# Processes using the GPU (do NOT touch these; vLLM is an existing workload)"
  run nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
else
  echo "nvidia-smi: NOT FOUND (this is not the DGX Spark, or the driver is missing)"
fi
have nvcc && run nvcc --version

section "Existing workloads (inspect only)"
echo '$ ps -ef | grep -E "[v]llm|[d]eepstream"'
ps -ef | grep -E '[v]llm|[d]eepstream' || echo "(none found)"

section "Docker"
if have docker; then
  run docker --version
  run docker compose version
  have nvidia-ctk && run nvidia-ctk --version
  run docker info --format 'runtimes={{json .Runtimes}} default={{.DefaultRuntime}}'
  run docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
  run docker image ls --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}'
else
  echo "docker: NOT FOUND"
fi

section "Project models and sample videos (read-only listing)"
for dir in deepstream/models data/media .; do
  [ -d "$dir" ] && find "$dir" -maxdepth 3 \( -iname '*.pt' -o -iname '*.onnx' -o -iname '*.engine' \
      -o -iname '*.mp4' \) -not -path '*/node_modules/*' -not -path '*/.git/*' 2>/dev/null | head -50
done
have ffprobe && echo "ffprobe present" || echo "ffprobe: NOT FOUND"

if [ "$CONTAINER_TEST" -eq 1 ]; then
  section "DeepStream container checks ($IMAGE)"
  if ! have docker; then echo "docker missing; skipped"; exit 0; fi
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    if [ "$PULL" -eq 1 ]; then run docker pull "$IMAGE"
    else echo "image not present locally; re-run with --pull to allow downloading it"; exit 0; fi
  fi
  RUN=(docker run --rm --runtime=nvidia --gpus all
       -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video "$IMAGE")
  echo "# GPU visible inside the container (a 'GB10 not yet supported' warning is documented as harmless)"
  run "${RUN[@]}" nvidia-smi -L
  run "${RUN[@]}" bash -lc 'deepstream-app --version-all 2>&1 | head -20'
  run "${RUN[@]}" bash -lc 'dpkg -l | grep -Ei "tensorrt|libnvinfer[0-9]* |cuda-toolkit|gstreamer1.0-tools " | head'
  run "${RUN[@]}" bash -lc 'python3 --version; python3 -c "import pyds; print(\"pyds\", pyds.__file__)" 2>&1 | tail -1; python3 -c "import gi; print(\"gi ok\")" 2>&1 | tail -1; python3 -c "import pyservicemaker; print(\"pyservicemaker ok\")" 2>&1 | tail -1'
  run "${RUN[@]}" bash -lc 'gst-inspect-1.0 nvv4l2decoder 2>&1 | head -3; gst-inspect-1.0 nvstreammux 2>&1 | head -3; gst-inspect-1.0 nvinfer 2>&1 | head -3; gst-inspect-1.0 nvtracker 2>&1 | head -3'
  run "${RUN[@]}" bash -lc 'ls -l /opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so /opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_*.yml 2>&1 | head; ls /opt/nvidia/deepstream/deepstream/sources/includes/nvdsinfer_custom_impl.h; ls /usr/src/tensorrt/bin/trtexec 2>&1'
fi
echo; echo "# done (read-only; nothing was modified)"
