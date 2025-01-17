#!/bin/bash

if [ "$ompi_test" == "yes" ]; then
    export PATH=$HOME/bin:$PATH
    export LD_LIBRARY_PATH=$HOME/lib:$LD_LIBRARY_PATH
    export MANPATH=$HOME/share/man:$MANPATH
fi

export UCX_IB_SL=1

if [ "$cuda" == "no" ]; then
    export CUDA_VISIBLE_DEVICES=""
    export OMPI_MCA_btl="^smcuda"
    export OMPI_MCA_mpi_cuda_support=0
fi

export RUN=srun
export RUNFLAGS=
export RES_DIR=$HOME/Swing_Test/results/
export BIN_DIR=$HOME/Swing_Test/bin
export RULE_FILE_PATH=$HOME/Swing_Test/ompi_rules/collective_rules.txt

export DEBUG_EXEC=$BIN_DIR/debug
export TEST_EXEC=$BIN_DIR/test
export RULE_UPDATER_EXEC=$BIN_DIR/change_collective_rules

export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
