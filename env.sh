# Source this from the calling shell to set some generally useful variables

# Tune thread usage for MKL et al
# Put these in an appropriate -rc file to turn on reliably
export KMP_AFFINITY='granularity=fine,compact,1,0'
export KMP_BLOCKTIME=1

# The corresponding OMP env variables (for non-MKL libraries: pip numpy, etc)
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))
export OMP_PROC_BIND='true'
export OMP_PLACES='cores'

# If conda has installed this lib then make sure it actually gets used
if [ -e "$CONDA_PREFIX/lib/libiomp5.so" ]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libiomp5.so"
fi
