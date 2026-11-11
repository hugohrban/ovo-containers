FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# Optionally rewrite the apt source lists at build time, e.g.
# --build-arg APT_SED='s|http://archive.ubuntu.com/ubuntu|https://<mirror-ubuntu>|g'
# You can chain multiple expressions with `;`
ARG APT_SED=""
RUN if [ -n "$APT_SED" ]; then \
        find /etc/apt/sources.list /etc/apt/sources.list.d/ -type f 2>/dev/null | xargs -r sed -i "$APT_SED" ; \
    fi

# Trust additional network certificates
ARG CERT_URL=""
RUN if [ -n "$CERT_URL" ]; then \
        apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \
        rm -rf /var/lib/apt/lists/* && \
        wget -O - "$CERT_URL" >> /etc/ssl/certs/ca-certificates.crt; \
    fi

RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    && rm -rf /var/lib/apt/lists/*

# esm hard-depends on cuequivariance-ops-torch-cu13 and the default PyPI torch
# wheel is built for CUDA 13. To target CUDA 12 GPUs we pull torch from the
# PyTorch cu12x index and use the cueq12 extra for the cu12 cuequivariance ops.
# Extras only add packages, so an override with a never-matching marker is
# still needed to drop the hard cu13 dependency.
# cu126 is the oldest PyTorch CUDA build available for torch 2.11; it runs on
# CUDA 12.x drivers >= 525 (e.g. 12.2 / 535) via CUDA minor version compatibility.
ARG TORCH_BACKEND=cu126
ENV UV_SYSTEM_CERTS=1
RUN pip install --no-cache-dir uv \
    && echo "cuequivariance-ops-torch-cu13 ; sys_platform == 'never'" > /tmp/overrides.txt \
    && uv pip install --system --no-cache \
        --torch-backend "$TORCH_BACKEND" \
        --override /tmp/overrides.txt \
        "esm[cueq12]" \
    && rm /tmp/overrides.txt

ENTRYPOINT []
