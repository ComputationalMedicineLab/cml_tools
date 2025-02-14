# XXX: No hashbang line, you are supposed to source this file from the calling
# shell!  Before running, make sure that $MAMBA_ROOT_PREFIX/.mambarc contains
# at least the channels specified in the included ./.mambarc

micromamba create --name='cml_tools' python=3.12.7 pip setuptools wheel
micromamba activate cml_tools

# Make sure that we have MKL BLAS pinned for the following installs
echo 'libblas=*=*mkl' >> $MAMBA_ROOT_PREFIX/envs/cml_tools/conda-meta/pinned

## By line:
## Other miscellaneous dev tools (install, build, profile, etc)
## Basic numeric python stack
## MKL and other intel related packages
## scikit-learn and pytorch packages (remove cpuonly if on machine with GPUs)...
## ... and various database tools
micromamba install \
    ninja make cmake pkg-config gcc gxx gdb valgrind line_profiler \
    numpy scipy pandas sympy cython matplotlib seaborn jupyter ipython \
    mkl mkl-devel mkl-include intel-openmp ipp ipp-devel ipp-include \
    scikit-learn pytorch cpuonly pyodbc psycopg2 databricks-sql-connector

# Version checks
set -x
gcc --version
c++ --version
python --version
jupyter --version
python -c 'import cython;print(cython.__version__)'
MKL_VERBOSE=1 python -c 'import numpy;numpy.__config__.show()'
MKL_VERBOSE=1 python -c 'import scipy;print(scipy.__version__)'
MKL_VERBOSE=1 python -c 'import sklearn;print(sklearn.__version__)'
MKL_VERBOSE=1 python -c 'import torch;print(torch.__version__);print(torch.__config__.show())'
set +x

# Tune thread usage for MKL et al
# Put these in an appropriate -rc file to turn on reliably
export KMP_AFFINITY='granularity=fine,compact,1,0'
export KMP_BLOCKTIME=1
