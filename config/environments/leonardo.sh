# Variables always needed
export SWINGCC=mpicc
export RUN=srun
export SWING_DIR=$HOME/Swing_Test

# Account/partition specific variables
export PARTITION=boost_usr_prod
# export ACCOUNT=IscrC_ASCEND
if [[ "$PARTITION" == "boost_usr_prod" ]]; then
    export GPU_NODE_PARTITION=4
    export CPU_NODE_PARTITION=32

    if [[ "$N_NODES" -gt 256 ]]; then
        export ACCOUNT=IscrB_SWING
        export QOS='qos_special'
        # export QOS_TASKS_PER_NODE=32
        # export QOS_GRES='gpu:4'
    elif [[ "$N_NODES" -gt 64 ]]; then
        export QOS='boost_qos_bprod'
        export QOS_TASKS_PER_NODE=32 # necessary for the qos
        export QOS_GRES='gpu:4'
    fi

    [[ "$N_NODES" == 2 && "$DEBUG_MODE" == "yes" ]] && export QOS='boost_qos_dbg'
fi

export EXCLUDE_NODES='lrdn0031,lrdn0032,lrdn0033,lrdn0034,lrdn0035,lrdn0036,lrdn0037,lrdn0038,lrdn0043,lrdn0044,lrdn0045,lrdn0046,lrdn0051,lrdn0052,lrdn0053,lrdn0054,lrdn0055,lrdn0056,lrdn0057,lrdn0058,lrdn0060,lrdn0139,lrdn0156,lrdn0201,lrdn0216,lrdn0304,lrdn0393,lrdn0393,lrdn0938,lrdn0952,lrdn1338,lrdn1450,lrdn1641,lrdn1653,lrdn1743,lrdn1809,lrdn1810,lrdn1811,lrdn1812,lrdn1817,lrdn1818,lrdn1819,lrdn1820,lrdn1829,lrdn1830,lrdn2257,lrdn2364'

export UCX_IB_SL=1
# export UCX_MAX_RNDV_RAILS=4
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
