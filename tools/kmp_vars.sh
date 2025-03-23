#!/bin/bash
# Intel-OpenMP options corresponding to the OpenMP variables ('./omp_vars.sh')
# They can conflict in difficult to reproduce ways if both sets of vars are set
unset OMP_NUM_THREADS
unset OMP_PLACES
unset OMP_PROC_BIND

export KMP_AFFINITY='granularity=fine,compact,1,0'
export KMP_BLOCKTIME=0
export LD_PRELOAD="$CONDA_PREFIX/lib/libiomp5.so:$LD_PRELOAD"
