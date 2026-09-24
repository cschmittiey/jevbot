# syntax=docker/dockerfile:1
# The bot, on distroless. It deliberately needs no torch: Laya runs as its own service
# (see docker-compose.yml) and is reached over HTTP, which is what keeps this image small.
ARG PYTHON_IMAGE=python:3.14-slim-bookworm
ARG RUNTIME_IMAGE=gcr.io/distroless/python3-debian12:nonroot

FROM ${PYTHON_IMAGE} AS build
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# pip --target instead of a virtualenv: a venv's interpreter symlink points at the
# builder's /usr/local/bin/python3.11, which does not exist on distroless, so the venv
# arrives broken. PYTHONPATH sidesteps it, and 3.11 here matches 3.11 there.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --target /pylibs -r /tmp/requirements.txt \
    && PYTHONPATH=/pylibs python -c "import discord, aiohttp, dotenv; print('deps ok')"

FROM ${RUNTIME_IMAGE}
COPY --from=build /pylibs /pylibs
WORKDIR /app
COPY jev_bot.py vocab.txt /app/
ENV PYTHONPATH=/pylibs \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JEV_BACKEND=laya \
    LAYA_URL=http://laya-serve:8000
# Distroless has no shell, so invoke the interpreter directly. Runs as the image's
# nonroot user (65532); no secrets are baked in -- see deploy/jevbot.env.example.
ENTRYPOINT ["/usr/bin/python3", "/app/jev_bot.py"]
