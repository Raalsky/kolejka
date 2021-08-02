FROM nvcr.io/nvidia/pytorch:19.12-py3

RUN /usr/bin/env conda install -y numba==0.53.1

COPY ./reserve_gpu_memory.py /app/
