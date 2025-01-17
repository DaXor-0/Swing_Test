#!/bin/bash

if [ "$ompi_test" == "yes" ]; then
    source ~/use_ompi.sh
fi

export RUN=mpiexec
export RUNFLAGS=
export RES_DIR=./local_results/
export BIN_DIR=./bin
export RULE_FILE_PATH=$HOME/University/Tesi/test/ompi_rules/collective_rules.txt

export DEBUG_EXEC=$BIN_DIR/debug
export TEST_EXEC=$BIN_DIR/test
export RULE_UPDATER_EXEC=$BIN_DIR/change_collective_rules

export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
