# Variables always needed
export SWINGCC=mpicc
export RUN=srun
export SWING_DIR=$HOME/Swing_Test

# Account/partition specific variables
export PARTITION=boost_usr_prod
if [[ "$PARTITION" == "boost_usr_prod" ]]; then
    export GPU_NODE_PARTITION=4

    if [[ "$N_NODES" -gt 64 ]]; then
        export QOS='boost_qos_bprod'
        export QOS_TASKS_PER_NODE=32 # necessary for the qos
        export QOS_GRES='gpu:4'
    fi

    [[ "$N_NODES" == 2 && "$DEBUG_MODE" == "yes" ]] && export QOS='boost_qos_dbg'
fi
export ACCOUNT=IscrC_ASCEND

export UCX_IB_SL=1
export MODULES="python/3.11.6--gcc--8.5.0"

[[ "$CUDA" == "True" ]] && export MODULES="cuda/12.1,$MODULES"

# MPI library specific variables
export MPI_LIB='OMPI'    # Possible values: OMPI, OMPI_SWING (beware that OMPI_SWING must be manually installed in the home directory)
if [ "$MPI_LIB" == "OMPI_SWING" ]; then
    export PATH=$HOME/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
    export MANPATH=$HOME/share/man:$MANPATH
    export MPI_LIB_VERSION='5.0.0'
else
    export MPI_LIB_VERSION='4.1.6'
    export MODULES="openmpi/4.1.6--gcc--12.2.0,$MODULES"
fi

# Load test dependnt environment variables
load_other_env_var(){
    export OMPI_MCA_coll_hcoll_enable=0
    export OMPI_MCA_coll_tuned_use_dynamic_rules=1
    if [ "$CUDA" == "False" ]; then
        export OMPI_MCA_btl="^smcuda"
        export OMPI_MCA_mpi_cuda_support=0
    else
        export OMPI_MCA_btl=""
        export OMPI_MCA_mpi_cuda_support=1
    fi
}
export -f load_other_env_var
