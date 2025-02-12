#!/bin/bash

# Variables always needed
export CC=mpicc
export RUN=srun
export RUNFLAGS=--mpi=pmix
export SWING_DIR=$HOME/Swing_Test/

# TODO: test and debug this
load_python() {
    warning "THIS NEEDS TO BE CHECKED"
    module load python || { error "Failed to load Python module." ; return 1; }
    return 0
}

# Load environment variables dependant on the MPI library
load_other_env_var() {
    if [ "$MPI_LIB" == "OMPI_SWING" ]; then
        export PATH=$HOME/bin:$PATH
        export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
        export MANPATH=$HOME/share/man:$MANPATH
    fi

    if [[ "$MPI_LIB" == "OMPI_SWING" ]] || [[ "$MPI_LIB" == "OMPI" ]]; then
        export OMPI_MCA_coll_hcoll_enable=0
        export OMPI_MCA_coll_tuned_use_dynamic_rules=1
        if [ "$CUDA" == "False" ]; then
            export CUDA_VISIBLE_DEVICES=""
            export OMPI_MCA_btl="^smcuda"
            export OMPI_MCA_mpi_cuda_support=0
        fi
    fi
}

