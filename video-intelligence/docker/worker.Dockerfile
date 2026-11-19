# GPU worker image, based on NVIDIA's DeepStream SBSA image for DGX Spark.
# The tag MUST be verified against your driver/CUDA first (scripts/env_check.sh --container-test).
ARG DEEPSTREAM_IMAGE=ghcr.io/nvidia/deepstream:9.2-triton-sbsa-dgx-spark
FROM ${DEEPSTREAM_IMAGE}

ENV USE_NEW_NVSTREAMMUX=yes \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends python3-venv g++ make ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# The DeepStream image deletes some codec libraries (libvpx, libmp3lame, libFLAC, ...) while
# their packages stay registered as installed, so ffprobe exited 127 on the Spark
# ("libvpx.so.9: cannot open shared object file"). Find every missing library with ldd, map
# it to its package with dpkg -S, reinstall those packages, and repeat until none is missing.
# The final `ffprobe -version` fails the build loudly if that did not work (it needs no GPU).
RUN set -eu; apt-get update; \
    for pass in 1 2 3 4; do \
      missing="$(ldd /usr/bin/ffprobe | awk '/not found/ {print $1}' | sort -u)"; \
      [ -z "$missing" ] && break; \
      echo "pass $pass missing: $missing"; \
      pkgs="$(for lib in $missing; do dpkg -S "$lib" 2>/dev/null | head -n1 | cut -d: -f1; done | sort -u)"; \
      echo "reinstalling: $pkgs"; \
      [ -n "$pkgs" ] || { echo "no package owns: $missing" >&2; exit 1; }; \
      apt-get install -y --no-install-recommends --reinstall $pkgs; \
    done; \
    ffprobe -version | head -n1; ffprobe -version >/dev/null; \
    rm -rf /var/lib/apt/lists/*

# --system-site-packages so the container's GStreamer bindings (gi) and pyds are visible.
RUN python3 -m venv --system-site-packages /opt/venv

# pyds (DeepStream Python bindings) is NOT shipped in the 9.2 image and has no prebuilt wheel for
# 9.x. NVIDIA's own script builds it from source (SBSA-aware); verified on the DGX Spark to yield
# pyds 1.2.3. Needs outbound HTTPS to github.com at BUILD time; takes ~10-20 min, then is cached.
# The built module lives in the script's venv, so expose it to /opt/venv through a .pth file.
RUN cd /opt/nvidia/deepstream/deepstream && ./user_deepstream_python_apps_install.sh --build-bindings
RUN set -eux; \
    PYDS_SO="$(find /opt/nvidia/deepstream/ -path '*site-packages*' -name 'pyds*.so' | head -n1)"; \
    echo "found: $PYDS_SO"; test -n "$PYDS_SO"; \
    dirname "$PYDS_SO" > /opt/venv/lib/python3.12/site-packages/pyds_path.pth; \
    cat /opt/venv/lib/python3.12/site-packages/pyds_path.pth
# NOTE: do not `import pyds` here. It links libcuda.so.1, which only exists at container run time
# (NVIDIA runtime injects it); `docker build` has no GPU. Verify at run time instead:
#   docker run --rm --runtime=nvidia --gpus all --entrypoint /opt/venv/bin/python \
#     video-intelligence-worker -c "import pyds; print(pyds.__file__)"

WORKDIR /app
COPY pyproject.toml ./
COPY shared ./shared
COPY services ./services
RUN /opt/venv/bin/pip install --no-cache-dir .

# DeepStream runner, nvinfer/tracker configs and the custom YOLO parser (compiled in-image so
# it matches the container's DeepStream headers; a failed build fails the image build loudly).
COPY deepstream/app /opt/vi/deepstream/app
COPY deepstream/configs /opt/vi/deepstream/configs
COPY deepstream/parser /opt/vi/parser
COPY deepstream/models/labels.txt /opt/vi/models/labels.txt
COPY scripts /app/scripts
RUN chmod +x /app/scripts/*.sh /app/scripts/*.py && make -C /opt/vi/parser

CMD ["/opt/venv/bin/python", "-m", "services.worker.main"]
