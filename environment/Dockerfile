# hash:sha256:fe2ad10f4615235592d11151b916ac9f97ba5c171fd8a24be87dfe72f8c74d02
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    spikeinterface[full]==0.102.3 \
    wavpack-numcodecs==0.2.2
