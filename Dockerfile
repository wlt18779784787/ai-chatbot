FROM python:3.12-slim

ARG APT_MIRROR_HOST=mirrors.aliyun.com
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
ENV PIP_TRUSTED_HOST=mirrors.aliyun.com

WORKDIR /app

RUN sed -i "s|deb.debian.org|${APT_MIRROR_HOST}|g; s|security.debian.org|${APT_MIRROR_HOST}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data /app/runtime

EXPOSE 9090

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "9090"]
