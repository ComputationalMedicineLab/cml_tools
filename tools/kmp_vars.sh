#!/bin/bash
# Intel-OpenMP options corresponding to the OpenMP variables ('./omp_vars.sh')
# They can conflict in difficult to reproduce ways if both sets of vars are set
unset OMP_NUM_THREADS
unset OMP_PLACES
unset OMP_PROC_BIND

export KMP_AFFINITY='granularity=fine,compact,1,0'
export KMP_BLOCKTIME=0

# Preloading Intel-OpenMP sometimes seems to help performance; certainly,
# compiling an application with MKL against iomp will easily double performance
# on machines with (at least some families of) Intel CPUs. Unfortunately, using
# ld preload as a workaround for not compiling against iomp seems to cause
# torch to segfault where an IndexError or similar exception should be thrown:
#
#   >>> torch.rand(100)[101]  # segfaults
#
# So it seems the only reliable way to get Intel-OpenMP's performance boost
# will be by compiling torch ourselves. Although this worked for a long time
# (or at least wasn't obviously broken), as of March 2025 the ABI is
# incompatible or something, and this LD_PRELOAD should not be used.
#
# See also ./check_segfault.py and ./check_tanh.py
#export LD_PRELOAD="$CONDA_PREFIX/lib/libiomp5.so"
