#!/bin/bash
# OpenMP variables corresponding to the KMP variables ('./kmp_vars.sh')
# They can conflict in difficult to reproduce ways if both sets of vars are set
unset KMP_AFFINITY
unset KMP_BLOCKTIME

export OMP_NUM_THREADS=$(( $(nproc) / 2 ))
export OMP_PROC_BIND='true'
export OMP_PLACES='cores'
