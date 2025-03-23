#!/bin/bash
# If running this script ('./build_env.sh') fails because micromamba cannot be
# found but micromamba is available from your calling shell, source this script
# instead ('source build_env.sh').
micromamba create -f ./base_environment.yaml

# Activating cml_tools gets correct $CONDA_PREFIX into the script env
micromamba activate cml_tools

# Make sure that we have MKL BLAS pinned for any later installs via micromamba
# Comment this out on non-Intel machines
echo 'libblas=*=*mkl' >> "$CONDA_PREFIX/conda-meta/pinned"

# Verify the python and pip versions and locations, then install via pip
python --version
pip --version
pip cache purge
pip install -r ./requirements.txt

# Automatically (un)set environment variables on (de)activate
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d/"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d/"
# Change which of the below is commented as needed for OMP vs KMP installs
cat kmp_vars.sh > "$CONDA_PREFIX/etc/conda/activate.d/set_vars.sh"
#cat omp_vars.sh > "$CONDA_PREFIX/etc/conda/activate.d/set_vars.sh"
source "$CONDA_PREFIX/etc/conda/activate.d/set_vars.sh"

# Verify the installations
# Test if MKL is installed and activated
MKL_VERBOSE=1 python -c 'import numpy, torch'

# Check versions of major dependencies
python << EOF
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
