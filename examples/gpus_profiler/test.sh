#!/bin/sh
EXAMPLE="gpus"

echo "###############"
echo "KOLEJKA EXAMPLE"
echo "System: $(whoami)@$(hostname):$(pwd)"
echo "Date: $(date)"
echo "Example: ${EXAMPLE}"
echo "###############"
echo ""

call() {
    echo "#>" "$@"
    "$@"
    res="$?"
    echo ""
    return "${res}"
}

call env
call python3 --version
call python3 -c 'import numba; print(numba.__version__)'
call python3 -c 'import numpy; print(numpy.__version__)'
call nvprof -o profile.nprof python3 sample.py
