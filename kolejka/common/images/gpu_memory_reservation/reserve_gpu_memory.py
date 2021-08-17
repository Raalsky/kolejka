import sys
import logging

import numpy as np
from numba import cuda
from numba.cuda.cudadrv.driver import CudaAPIError, Device


ARRAY_ELEMENT_DTYPE = np.uint8
ARRAY_ELEMENT_SIZE = np.dtype(ARRAY_ELEMENT_DTYPE).itemsize


class NotEnoughMemory(RuntimeError):
    def __init__(
            self,
            gpu: Device,
            bytes_to_preserve: int,
            bytes_free: int,
            bytes_total: int
    ):
        super().__init__()
        logging.error(
            f"Out of memory on GPU#{gpu.id}({str(gpu.name, 'utf-8')}) when trying to allocate {bytes_to_preserve} bytes."
            f" Before running available memory was {bytes_free} out of {bytes_total} total."
        )


def mb_to_bytes(mbytes: int) -> int:
    return mbytes * 1024**2


def main(desired_memory: int) -> None:
    """
    Expected number of bytes to be free in all available GPUs memory
    :param desired_memory:
    """
    preserved_memory = {}
    desired_memory = mb_to_bytes(desired_memory)

    for gpu_index, gpu in enumerate(cuda.gpus.lst):
        with gpu:
            # Initialize CUDA context preserves minor amount of memory to be allocated
            _ = cuda.device_array((1,))

            # Retrieve current device free memory space (in bytes)
            bytes_free, bytes_total = cuda.current_context().get_memory_info()

            bytes_to_preserve = bytes_free - desired_memory

            if bytes_to_preserve < 0:
                raise NotEnoughMemory(gpu, desired_memory, bytes_free, bytes_total)

            if bytes_to_preserve > 0:
                try:
                    preserved_memory[gpu_index] = cuda.device_array(
                        (bytes_to_preserve // ARRAY_ELEMENT_SIZE,),
                        dtype=ARRAY_ELEMENT_DTYPE
                    )
                    logging.debug(
                        f"Preserved {bytes_to_preserve} bytes on GPU#{gpu.id}({str(gpu.name, 'utf-8')})"
                        f" using array of size {preserved_memory[gpu_index].size}"
                        f" with type of {preserved_memory[gpu_index].dtype}"
                    )
                except CudaAPIError as e:
                    logging.error(f"An error occurred on GPU#{gpu.id}({str(gpu.name, 'utf-8')}): {e}")
                    exit(-1)

    while True:
        try:
            pass
        except KeyboardInterrupt:
            exit()


if __name__ == '__main__':
    """
    Usage python reserve_gpu_memory.py <MiB-TO-RESERVE>
    """
    logging.basicConfig(level=logging.DEBUG)
    main(int(sys.argv[-1]))
