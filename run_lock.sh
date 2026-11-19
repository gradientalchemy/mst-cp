#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
PY=${PY:-python}
date > run_lock.start
TAG=LOCKA NP=12 $PY mst.py 3,4,5 > lock_A.log 2>&1
TAG=LOCKB NP=7  $PY mst.py 3,4,5 > lock_B.log 2>&1
TAG=LOCK NP=12 $PY mst2.py 3,4,5 > lock_v11R.log 2>&1
TAG=LOCK ORDERED=1 FWD=1 NP=12 $PY mst2.py 3,4,5 > lock_v11O.log 2>&1
date > run_lock.done
