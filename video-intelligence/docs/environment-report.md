# Environment report (Phase 1)

**Status: Phase 1 was NOT executed on the DGX Spark.** The session was linked to a MacBook Pro
(macOS, arm64) with an empty project folder, not to the DGX Spark. Everything below is
therefore split into three clearly separated kinds of statement:

| Kind | Meaning |
|---|---|
| **Observed** | Seen by commands I ran, in the authoring sandbox (not the Spark) |
| **Documented** | Taken from NVIDIA documentation (links at the bottom) |
| **Pending on Spark** | Must be confirmed by running `scripts/env_check.sh` on the DGX Spark |

Nothing here may be read as a measurement of the Spark.

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

## 5. Checklist to complete Phase 1 on the DGX Spark

Run `scripts/env_check.sh --container-test` and paste the output below this line. Phase 1 is
complete only when all of these are true:

- [ ] `nvidia-smi` shows GB10 and the driver; vLLM processes are listed and **unchanged**
- [ ] `docker info` lists the `nvidia` runtime; `docker run --gpus all ... nvidia-smi -L` works
- [ ] DeepStream image pulled; `deepstream-app --version-all` prints versions (record them here)
- [ ] `python3 -c "import pyds"` works in the image (else see risk 2)
- [ ] `gst-inspect-1.0 nvv4l2decoder nvstreammux nvinfer nvtracker` all succeed
- [ ] Tracker library and sample tracker YAML exist at the configured paths
- [ ] `/opt/nvidia/deepstream/deepstream/sources/includes/nvdsinfer_custom_impl.h` exists
- [ ] MemAvailable noted before and after a run, to set `GPU_MIN_AVAILABLE_MEM_MB`

### Output from the DGX Spark

_Not yet collected._

## Sources

- DeepStream installation (9.2): https://docs.nvidia.com/deepstream/text/DS_Installation
- DeepStream installation (9.1): https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Installation.html
- DeepStream Docker containers (9.1): https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_docker_containers.html
- DeepStream release notes: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Release_notes.html
- Custom model / nvinfer parser interface: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_using_custom_model.html
