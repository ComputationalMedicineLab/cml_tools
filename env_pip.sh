# XXX: No hashbang line, you are supposed to source this file from the calling
# shell!  Before running, make sure that $MAMBA_ROOT_PREFIX/.mambarc contains
# at least the channels specified in the included ./.mambarc
micromamba create --name='cml_tools' python=3.12 pip setuptools wheel
micromamba activate cml_tools

# Verify the python and pip versions and locations
python --version
pip --version
pip cache purge

# Make sure that we have MKL BLAS pinned for later installs
echo 'libblas=*=*mkl' >> "$CONDA_PREFIX/conda-meta/pinned"

# Automatically set some environment variables on activate
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d/"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d/"

cat >"$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" << EOF
#!/bin/bash
export KMP_AFFINITY='granularity=fine,compact,1,0'
export KMP_BLOCKTIME=1
export LD_PRELOAD="$CONDA_PREFIX/lib/libiomp5.so"
EOF

cat >"$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh" << EOF
unset LD_PRELOAD
EOF

# Run the above env var scripts
source "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

# The first is CPU only, the second is for Cuda 12.6 - change as appropriate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install numpy scipy pandas sympy cython matplotlib seaborn jupyter ipython \
    scikit-learn pyodbc psycopg2-binary databricks-sql-connector intel-openmp

# Post-install version checks
jupyter --version
ipython --version
MKL_VERBOSE=1 python << EOF
import cython, numpy, scipy, sklearn, torch
sep = ("-"*80)+"\n"
print(f'{sep}Numpy version: {numpy.__version__}')
print(numpy.show_config())
print(f'{sep}Cython version: {cython.__version__}')
print(f'{sep}scipy version: {scipy.__version__}')
print(f'{sep}sklearn version: {sklearn.__version__}')
print(f'{sep}torch version: {torch.__version__}')
print(torch.__config__.show())
print(sep)
EOF
