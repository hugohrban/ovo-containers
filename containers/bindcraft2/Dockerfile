# BindCraft2 (https://github.com/PacesaLab/BindCraft2), based on its containers/Dockerfile,
# built for CUDA 12 with the compilers capped for R535 (CUDA 12.2) drivers.
# AlphaFold parameters (5.3 GB) are not baked in, they are downloaded on the first campaign.
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
ENV PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# Trust additional network certificates
ARG CERT_URL=""
RUN if [ ! -z "$CERT_URL" ]; then curl "$CERT_URL" >> /etc/ssl/certs/ca-certificates.crt; fi
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Editable install: the runtime finds settings/ and scaffolds/ relative to the source tree
ARG BINDCRAFT2_REVISION=7a2dfdb
RUN git clone https://github.com/PacesaLab/BindCraft2.git /opt/bindcraft \
    && git -C /opt/bindcraft checkout "$BINDCRAFT2_REVISION" \
    && python3 -m pip install -e "/opt/bindcraft[cuda12]"

# R535 drivers can't load CUBINs from the 12.9 compilers, so cap nvcc/nvrtc below 12.3.
# nvjitlink stays at 12.9, otherwise libcusparse/libcublas fail to load.
RUN python3 -m pip install --no-deps --force-reinstall \
      "nvidia-cuda-nvcc-cu12<12.3" "nvidia-cuda-nvrtc-cu12<12.3" "nvidia-nvjitlink-cu12==12.9.86"

# The CUDA wheels keep their libraries under site-packages/nvidia/*/lib, where the loader doesn't look,
# and plain Ubuntu doesn't set the variables that make the NVIDIA container hook inject the driver
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
RUN python3 -c "import nvidia, pathlib; print('\n'.join(sorted(str(path) for root in nvidia.__path__ for path in pathlib.Path(root).glob('*/lib'))))" > /etc/ld.so.conf.d/bindcraft-cuda.conf \
    && ldconfig && ldconfig -p | grep -q libcupti

# Fail the build rather than the first campaign if the wheels don't line up
WORKDIR /
RUN python3 -c "import jax, bindcraft.proteinmpnn; print('jax', jax.__version__)" \
    && python3 -c "from ctypes import CDLL; CDLL('libcusparse.so.12'); CDLL('libcublas.so.12')" \
    && bindcraft --help > /dev/null \
    && python3 -m bindcraft.selfcheck cuda12 --shipped-only

WORKDIR /work
