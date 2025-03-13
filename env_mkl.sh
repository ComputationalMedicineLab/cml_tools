# XXX: No hashbang line, you are supposed to source this file from the calling
# shell!  Before running, make sure that $MAMBA_ROOT_PREFIX/.mambarc contains
# at least the channels specified in the included ./.mambarc

micromamba create --name='cml_tools_mkl' python=3.12 pip setuptools wheel
micromamba activate cml_tools_mkl

# Verify the python and pip versions and locations
python --version
pip --version
pip cache purge

# Make sure that we have MKL BLAS pinned for the micromamba installs
echo 'libblas=*=*mkl' >> $MAMBA_ROOT_PREFIX/envs/cml_tools/conda-meta/pinned

micromamba install \
    ninja make cmake pkg-config gcc gxx gdb valgrind line_profiler \
    numpy scipy pandas sympy cython matplotlib seaborn jupyter ipython \
    mkl mkl-devel mkl-include intel-openmp ipp ipp-devel ipp-include \
    scikit-learn pytorch-cpu pyodbc psycopg2 databricks-sql-connector

# Post-install version checks
gcc --version
c++ --version
jupyter --version
ipython --version
# Should be able to see MKL being loaded by numpy / scipy
MKL_VERBOSE=1 python -c 'import cython;print(cython.__version__)'
MKL_VERBOSE=1 python -c 'import numpy;numpy.__config__.show()'
MKL_VERBOSE=1 python -c 'import scipy;print(scipy.__version__)'
MKL_VERBOSE=1 python -c 'import sklearn;print(sklearn.__version__)'
MKL_VERBOSE=1 python -c 'import torch;print(torch.__version__);print(torch.__config__.show())'

# Tune thread usage for MKL et al
# Put these in an appropriate -rc file to turn on reliably
export KMP_AFFINITY='granularity=fine,compact,1,0'
export KMP_BLOCKTIME=1
