#!/bin/bash

# General variables always needed
export CC=cc
export RUN=srun
export SWING_DIR=$HOME/Swing_Test

# Account/partition specific variables
export PARTITION=standard
export QOS=''
export ACCOUNT=project_465000997

# MPI library specific variables
export MPI_LIB='CRAY_MPICH'
export MPI_LIB_VERSION='8.1.29'

module load LUMI
module load CrayEnv

# Used to load python and virtual environment
load_python() {
# It contains
# - python-3.11.5
# - numpy-1.24.4
# - scipy-1.10.1
# - mpi4py-3.1.4
# - dask-2023.6.1
    module load cray-python/3.11.5 || { error "Failed to load Python module." ; return 1; }
    return 0
}

# Dummy function to load other environment variables to silence errors
load_other_env_var() {
}
