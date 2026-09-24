# laya-serve: the decision engine jevbot talks to.
#
# Built from the PyPI package rather than a vendored checkout -- upstream publishes
# `laya[serve]` (Apache-2.0, https://laya.convaiinnovations.com) and ships no registry
# image. CPU wheels by default; build with TORCH_INDEX=cu128 for an NVIDIA host.
ARG PYTHON_IMAGE=python:3.11-slim-bookworm
FROM ${PYTHON_IMAGE}

ARG TORCH_INDEX=cpu
ARG TORCH_VERSION=2.14.0
ARG LAYA_VERSION=0.3.20

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/laya/.cache/huggingface \
    LAYA_HOST=0.0.0.0 \
    LAYA_PORT=8000 \
    LAYA_DEVICE=cpu \
    LAYA_THREADS=4 \
    OMP_NUM_THREADS=4

RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
    && pip install "laya[serve]==${LAYA_VERSION}" \
    && pip check

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin laya \
    && mkdir -p /home/laya/.cache/huggingface \
    && chown -R laya:laya /home/laya
USER 10001

EXPOSE 8000
# Model weights live in the mounted cache, so this downloads ~800MB on first boot only.
# start-period is long because LAYA_PRELOAD=1 loads checkpoints before the server binds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --start-interval=2s \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('LAYA_PORT','8000') + '/health', timeout=4)"

CMD ["laya-serve"]
