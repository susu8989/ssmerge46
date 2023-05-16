FROM python:3.10-slim-buster

RUN mkdir /app
WORKDIR /app

ENV PYTHONFAULTHANDLER=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y gdal-bin libgdal-dev libgl1-mesa-dev && \
    python -m pip install -U pip poetry

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false && \
  poetry install --no-interaction --no-root --no-dev && \
  rm -rf ~/.cache/pypoetry && \
  rm -rf ~/.config/pypoetry

COPY ./ /app/

CMD python run.py