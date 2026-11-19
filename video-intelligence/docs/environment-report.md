# Environment report (Phase 1)

**Status (2026-11-19): Phase 1 executed on the DGX Spark; see section 5 for measured results.**
Sections 1-4 were written before access to the Spark and are kept for context; where they
disagree with section 5, section 5 (measured on the machine) wins.

## 1. What was observed (authoring sandbox, NOT the DGX Spark)

| Item | Observed value |
|---|---|
| Architecture | x86_64 (the Spark is aarch64) |
| OS | Ubuntu 24.04.5 LTS |
| CPU / RAM | 2 vCPU, 7.8 GiB RAM, no swap |
| GPU | none (`nvidia-smi` not found) |
| Docker | client 29.10.0, Compose v5.7.1; **daemon not running**, so no container could be started |
| PostgreSQL | 16.x server binaries present: used for all database integration tests |
| RabbitMQ | not installable (no package, no Erlang): broker tests were **not run** |
| ffmpeg / ffprobe | present: used to generate and probe real MP4 test files |
| Python | 3.13 (project targets 3.12; no 3.13-only syntax is used) |
| Connected user folder | empty: **no sample videos, no YOLO weights** |

`scripts/env_check.sh` was run in this sandbox only to prove the script works; its output
(correctly) reports "nvidia-smi NOT FOUND".

## 2. User-provided baseline (not verified by me)

| Item | Value reported by the user |
|---|---|
| Device | NVIDIA DGX Spark GB10, aarch64, Ubuntu 24.04.5 LTS |
| Driver / CUDA | 580.178.04 / CUDA 13.0 (as `nvidia-smi` reports) |
| Memory | 128 GB unified (about 121 GiB visible, about 63 GiB available when last observed) |
| Docker / Compose / NVIDIA Container Toolkit | 29.6.2 / 5.2.0 / 1.20.1 |
| Storage | NVMe 3.7 TB |
| Existing workloads | 2 vLLM `EngineCore` processes, about 15 GB and 30 GB GPU memory when last reported |

**vLLM was not inspected and will not be touched.** Rules enforced by this project:
no stop / restart / reconfigure / kill of vLLM or any existing container; the GPU scheduler only
limits *our* concurrency (slot table + `MemAvailable` headroom) and never signals other
processes; Compose creates only its own network and volumes; no `prune`, no `down -v`.

## 3. DeepStream compatibility (from NVIDIA documentation)

| Finding | Source |
|---|---|
| DGX Spark is supported through the **SBSA Triton container only**; native install is unsupported | DS install guide |
| DS 9.2 image: `ghcr.io/nvidia/deepstream:9.2-triton-sbsa-dgx-spark` | DS 9.2 install guide |
| DS 9.1 image: `nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark`, doc states driver 580.159.03 is required for SBSA/DGX Spark (bundled with the image per the doc) | DS 9.1 install guide |
| Container warning "Detected NVIDIA GB10 GPU, which is not yet supported" is documented as harmless | DS release notes |
| VIC is not supported on DGX Spark: tiler needs `compute-hw=1`. Only `nv3dsink` is a supported display sink (we use `fakesink`) | DS 9.2 install guide / release notes |
| Recommended run flags: `--runtime=nvidia --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`; `--privileged` is not required | DS 9.2 install guide |
| `pyds` is **deprecated**; `pyservicemaker` is the recommended binding | DS release notes |
| DS 9.2 dGPU table lists CUDA 13.2, TensorRT 10.16.0.72, GStreamer 1.24.2 (the page gives **no SBSA/Spark-specific** versions) | DS 9.2 install guide |

### Open compatibility risks (decide on the Spark)

1. **CUDA 13.2 vs host driver "CUDA 13.0".** The 9.2 doc's dGPU table names CUDA 13.2 while the
   host driver reports 13.0. The doc does not say whether 9.2 SBSA runs on a 580.x driver. The
   only driver statement for the Spark is for **9.1 (580.159.03, older than the host's
   580.178.04)**. If 9.2 fails GPU checks, set `DEEPSTREAM_IMAGE=nvcr.io/nvidia/deepstream:9.1-triton-sbsa-dgx-spark`
   (one line in `.env`). Do not change the host driver.
2. **`pyds` may be missing in the image.** `ds_runner.py` uses `pyds` for the metadata probe and
   exits with code 4 (`runner_environment`) and a clear message if it cannot import it. The fix
   would be porting `FrameEmitter` to `pyservicemaker`; I did not write that because I could not
   confirm its API against a running container.
3. **Hardware decode on GB10 is not discussed** in the docs I read. `ds_runner.py` uses an
   explicit `qtdemux ! h264parse|h265parse ! nvv4l2decoder` path for H.264/H.265 so that
   hardware decode is verifiable (`gst-inspect-1.0 nvv4l2decoder`, `nvidia-smi dmon -s u` dec
   column during a run), and falls back to `uridecodebin` for other codecs.
4. **Timestamps.** `nvstreammux` may rewrite buffer PTS. `scripts/validate_timestamps.py`
   compares runner timestamps with `ffprobe` frames; the worker also rejects frames outside the
   segment window. Run it before trusting any result.
5. **Tracker sample config path** (`.../samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml`)
   and `libnvds_nvmultiobjecttracker.so` are the standard DeepStream locations but were not
   verified in the image; `env_check.sh --container-test` lists them.
6. **Custom parser** (`deepstream/parser`): the decode logic is compiled and unit-tested here;
   the DeepStream glue file has not been compiled against the real `nvdsinfer_custom_impl.h`
   (`docker build` of the worker image compiles it and fails loudly if it does not build).

## 4. YOLO model inventory and export strategy

**No YOLO model was available**: the connected folder was empty, and nothing could be inspected
on the Spark. The project status document records an earlier build on another path
(`~/Documents/feat_process_vdo/video-intelligence/`, YOLO11n + ByteTrack on CPU) that was not
connected to this session, so I could not check its weights.

Decision rule (implemented by `scripts/inspect_yolo_model.py` + two nvinfer configs):

| ONNX output (batch stripped) | Typical models | Config | Clustering |
|---|---|---|---|
| `[4+nc, anchors]` (e.g. `[84, 8400]`) | YOLOv8, YOLO11 | `config_infer_primary_yolo.txt` | `cluster-mode=2` (NMS in DeepStream) |
| `[300, 6]` (x1,y1,x2,y2,score,class) | YOLOv10, YOLO26 (end-to-end) | `config_infer_primary_yolo_e2e.txt` | `cluster-mode=4` (none) |

Flow: `export_yolo.py` (on the Spark) -> `inspect_yolo_model.py` confirms the layout and class
count -> set `num-detected-classes` -> `build_engine.sh` builds the FP16 TensorRT engine once.
`.engine` files must never be copied between machines.

## 5. Results measured on the DGX Spark (2026-11-19, host `spark-396a`)

Everything below was observed on the real machine; nothing is estimated.

### Phase 1 checklist

- [x] `nvidia-smi`: GB10, driver 580.178.04, CUDA 13.0. Two vLLM `EngineCore` processes
      (~15 GB and ~40.5 GB) were running throughout and were **not touched**
- [x] `nvidia` runtime present; `docker run --gpus all ... nvidia-smi -L` inside the DS image
      shows `NVIDIA GB10`
- [x] `ghcr.io/nvidia/deepstream:9.2-triton-sbsa-dgx-spark` pulled without login. The image
      runs in **CUDA forward-compatibility mode** (CUDA 13.2 user-mode driver 595.58.03 on the
      580.178.04 kernel driver), which closes open risk 1: the 9.1 fallback is not needed
- [x] TensorRT 10.16.1.11 (+cuda13.2), `trtexec` present
- [x] `gst-inspect-1.0`: `nvv4l2decoder`, `nvstreammux`, `nvinfer`, `nvtracker` present
- [x] `libnvds_nvmultiobjecttracker.so` and the sample tracker YAMLs (incl.
      `config_tracker_NvDCF_perf.yml`) exist at the configured paths (risk 5 closed)
- [x] `nvdsinfer_custom_impl.h` exists; the custom parser compiles in the worker image (risk 6
      closed for compilation; decoding verified by real detections, see below)
- [x] `pyds` is absent from the 9.2 image and has no prebuilt wheel for 9.x (latest release
      v1.2.2 targets DS 8.0). Built from source in the worker image via
      `user_deepstream_python_apps_install.sh --build-bindings` -> pyds 1.2.3 (risk 2 closed).
      `import pyds` needs `libcuda.so.1`, so it can only be checked at run time, not during
      `docker build`. `pyservicemaker` 0.0.1 is present but needs `pyyaml`
- [ ] MemAvailable before/after a run (to tune `GPU_MIN_AVAILABLE_MEM_MB`): not yet recorded

### Model

- The prebuilt Ultralytics engines (`yolo26n_fp16_b8/b16_640.engine`) were **not** used: they
  carry an Ultralytics metadata header, were built for batch 8/16 and for an unknown TensorRT
  version. The engine is built by nvinfer from ONNX inside the DS container instead.
- The first ONNX tried failed to build in TensorRT 10.16
  (`ISliceLayer /model.2/Split ... out of bounds access on axis 0`). Re-exporting `yolo26n.pt`
  with the `ultralytics/ultralytics:latest-nvidia-arm64` image
  (`imgsz=640 batch=1 dynamic=False simplify=True opset=17 half=False`) fixed it.
- FP16 engine `yolo_e2e.onnx_b1_gpu0_fp16.engine` built by nvinfer in **~14 s**, with
  `config_infer_primary_yolo_e2e.txt` (end-to-end layout, `cluster-mode=4`).
- Still to record: the output shape printed by `scripts/inspect_yolo_model.py`.

### Phase 2 pipeline on `output.mp4` (H.264 High, 2858x1350, 60 fps, 180.75 s, 10,845 frames)

| Check | Result |
|---|---|
| Decode path | `qtdemux ! h264parse ! nvv4l2decoder` (hardware path selected; `dec` utilisation not yet captured with `nvidia-smi dmon`) |
| Full file, one run | 10,845 frames emitted (= ffprobe `nb_frames`), pts 33 ms .. 180,766 ms, 9,611 frames with detections |
| Segment [60000, 70000) ms | 598 frames, 2 preroll frames dropped, `seek on demux` used, runner exit 0 |
| `validate_timestamps.py` (tolerance 2 ms) | ffprobe 600 / runner 598, ratio 0.997, **0 frames off by more than 2 ms -> TIMESTAMPS OK** (risk 4 closed for this file) |
| Wall time of that 10 s segment | 20.6 s including process start, engine deserialisation and pipeline setup. This is NOT a throughput benchmark |

Two of the 600 frames in the window are not emitted (ratio 0.997, within the 2% gate). Not yet
investigated; likely the first frames at the window start after the ACCURATE seek.

### Defects found and fixed on the first GPU runs

1. The DS image entrypoint re-splits arguments, breaking `docker run IMAGE bash -lc '...'`;
   use `--entrypoint bash` (fixed in `env_check.sh`).
2. `GLib.Error.domain` is already a `str`; `quark_to_string()` raised `TypeError` and masked
   the real pipeline error (fixed, unit-tested).
3. With `fakesink async=false` the state change never posts `ASYNC_DONE`; the runner waited
   120 s and gave up. It now waits on the pipeline state (fixed).
4. `nvstreammux` `printf`s (`max_fps_dur ...`) to fd 1 and corrupted the NDJSON stream. The
   runner now writes the protocol to a private duplicate of stdout and points fd 1 at stderr
   (fixed, unit-tested with a subprocess).
5. A pipeline-level seek is not forwarded through `nvstreammux`: a 10 s window decoded the
   whole 180 s file. The seek is now sent to `qtdemux`, and frames outside the window are
   dropped and counted (`dropped` in the summary) (fixed, unit-tested).
6. `ffprobe` in the worker image failed with `libvpx.so.9: cannot open shared object file`
   (the DS image strips codec libraries). The Dockerfile reinstalls `libvpx9` and checks
   `ffprobe -version` at build time (fix written; rebuild pending).

## Sources

- DeepStream installation (9.2): https://docs.nvidia.com/deepstream/text/DS_Installation
- DeepStream installation (9.1): https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html
- DeepStream Docker containers (9.1): https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_docker_containers.html
- DeepStream release notes: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html
- Custom model / nvinfer parser interface: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_using_custom_model.html
