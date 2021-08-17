FROM nvidia/cuda:11.0-base

RUN apt-get update && apt install -y python3 python3-pip
RUN pip3 install -q numba==0.53.1

COPY reserve_gpu_memory.py /app/
