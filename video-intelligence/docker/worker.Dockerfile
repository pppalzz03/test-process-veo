# GPU worker image, based on NVIDIA's DeepStream SBSA image for DGX Spark.
# The tag MUST be verified against your driver/CUDA first (scripts/env_check.sh --container-test).
ARG DEEPSTREAM_IMAGE=ghcr.io/nvidia/deepstream:9.2-triton-sbsa-dgx-spark
FROM ${DEEPSTREAM_IMAGE}

ENV USE_NEW_NVSTREAMMUX=yes \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends python3-venv g++ make ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# --system-site-packages so the container's GStreamer bindings (gi) and pyds are visible.
RUN python3 -m venv --system-site-packages /opt/venv

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
