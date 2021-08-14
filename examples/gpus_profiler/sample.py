import os
import math
import numpy as np
from numba import vectorize


@vectorize(['float32(float32)'], target='cuda')
def gpu_sqrt(x):
    return math.sqrt(x)


if __name__ == '__main__':
    a = np.arange(4096, dtype=np.float32)
    print(gpu_sqrt(a))
    # print('Hello World')
