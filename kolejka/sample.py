import os
import math
import numpy as np
from numba import vectorize


@vectorize(['float32(float32)'], target='cuda')
def gpu_sqrt(x):
    return math.sqrt(x)


if __name__ == '__main__':
    os.environ['NUMBAPRO_LIBDEVICE'] = "/usr/local/cuda-10.2/nvvm/libdevice"
    os.environ['NUMBAPRO_NVVM'] = "/usr/local/cuda-10.2/nvvm/lib64/libnvvm.so"

    a = np.arange(4096, dtype=np.float32)
    gpu_sqrt(a)
