# hash:sha256:4fd11e2b10fc65f0c85171b88795b7fa9a68669eab71e71c3d040f6190670104
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    spikeinterface[full]==0.103.0 \
    wavpack-numcodecs==0.2.2
