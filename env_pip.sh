# XXX: No hashbang line, you are supposed to source this file from the calling
# shell!  Before running, make sure that $MAMBA_ROOT_PREFIX/.mambarc contains
# at least the channels specified in the included ./.mambarc

micromamba create --name='cml_tools' python=3.12 pip setuptools wheel
micromamba activate cml_tools

# Verify the python and pip versions and locations
python --version
pip --version
pip cache purge

# The first is CPU only, the second is for Cuda 12.6 - change as appropriate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install numpy scipy pandas sympy cython matplotlib seaborn jupyter ipython \
    scikit-learn pyodbc psycopg2-binary databricks-sql-connector

# Post-install version checks
jupyter --version
ipython --version
# MKL_VERBOSE shouldn't do anything for any of these except maybe torch, but...
# doesn't hurt to check.
MKL_VERBOSE=1 python -c 'import cython;print(cython.__version__)'
MKL_VERBOSE=1 python -c 'import numpy;numpy.__config__.show()'
MKL_VERBOSE=1 python -c 'import scipy;print(scipy.__version__)'
MKL_VERBOSE=1 python -c 'import sklearn;print(sklearn.__version__)'
MKL_VERBOSE=1 python -c 'import torch;print(torch.__version__);print(torch.__config__.show())'

# Tune thread usage for MKL et al
# Put these in an appropriate -rc file to turn on reliably
# (maybe) still useful - pip numpy is OpenBLAS but pip torch may use MKL
export KMP_AFFINITY='granularity=fine,compact,1,0'
export KMP_BLOCKTIME=1
