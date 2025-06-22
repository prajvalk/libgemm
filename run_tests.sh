#!/bin/bash

set -e

export PYTHONPATH=/opt/pyscf-libgemm
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE LD_PRELOAD=$(pwd)/libgemm.so python3 run_tests.py 2>&1 | tee -a run_tests.out