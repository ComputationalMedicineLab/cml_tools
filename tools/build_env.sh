#!/bin/bash
# If running this script ('./build_env.sh') fails because micromamba cannot be
# found but micromamba is available from your calling shell, source this script
# instead ('source build_env.sh'). In either case, run or source from this dir.
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

# databricks-sql-connector has numpy pinned at 1.24 for some reason (it doesn't
# even use numpy), pandas has a similar version cap; there is a PR to fix:
# https://github.com/databricks/databricks-sql-python/pull/478#issue-2720301221
# In the meantime, we install databricks-sql-connector first and then upgrade
# numpy in requirements.txt
pip install databricks-sql-connector
pip install --upgrade -r ./requirements.txt
pip install --upgrade -r ./torch_requirements.txt

# Automatically (un)set environment variables on (de)activate
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d/"
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d/"
# Change which of the below is commented as needed for OMP vs KMP installs
cat kmp_vars.sh > "$CONDA_PREFIX/etc/conda/activate.d/set_vars.sh"
#cat omp_vars.sh > "$CONDA_PREFIX/etc/conda/activate.d/set_vars.sh"
cat unset_vars.sh > "$CONDA_PREFIX/etc/conda/deactivate.d/set_vars.sh"
source "$CONDA_PREFIX/etc/conda/activate.d/set_vars.sh"

# Verify the installations
# Test if MKL is installed and activated
MKL_VERBOSE=1 python -c 'import numpy, torch'
# Check versions of major dependencies
VERBOSE=1 python ./versions.py
