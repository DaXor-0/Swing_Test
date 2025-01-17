#!/bin/bash

if [ "$OMPI_TEST" == "yes" ]; then
    source ~/use_ompi.sh
fi

export RUN=mpiexec
export RUNFLAGS=

export SWING_DIR=$HOME/University/Tesi/test/

