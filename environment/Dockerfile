# hash:sha256:06a80167de348698240fa349eb15f0e7435418d26977e15d200f7a43b349c32f
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    spikeinterface[full]==0.101.1 \
    wavpack-numcodecs==0.2.1
