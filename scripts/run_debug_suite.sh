#!/bin/bash

cleanup() {
    echo "Caught Ctrl+C! Stopping the script and killing all child processes..."
    pkill -P $$    # Kills all processes whose parent is the current script
    exit 1         # Exit the script with a non-zero status
}

trap cleanup SIGINT

N_NODES=$1

location='local'
# location='leonardo'
# location='snellius'

# cuda=yes
cuda=no

# Use custom Open MPI build with swing defined inside library (it must be built with --prefix=$HOME
ompi_test='yes'
#ompi_test='no'

# Load the environment-specific configuration
if [ -f scripts/environments/${location}.sh ]; then
    source scripts/environments/${location}.sh
else
    echo "ERROR: Environment script for location '${location}' not found!"
    exit 1
fi

ALGOS=(8 9 10 11 12 13 14 15 16)
SKIP=(9 10 11 12 13 16)

ARR_SIZES=(8 64 512 2048 16384)
TYPES=('int64' )
# TYPES=('int32' 'int64' 'float' 'double' 'char' 'int8' 'int16')
# NOTE: problems with char, int8, int16

make clean
make all


# Test custom algorithms here, loop through:
# - algorithms:
#     - 8 swing latency
#     - 9 swing bandwidt memcp
#     - 10 swing bandwidth datatype
#     - 11 swing bandwidth datatype + memcp
#     - 12 swing bandwidth segmented
#     - 13 swing bandwidth static
#     - 14 swing latency OVER MPI
#     - 15 recursive doubling OVER MPI
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
#
# note that algo 14 and 15 are not defined in Opmi so
# ompi will go with default algorithm selection
for algo in ${ALGOS[@]}; do
    $RULE_UPDATER_EXEC $RULE_FILE_PATH $algo
    export OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_PATH}
    for size in "${ARR_SIZES[@]}"; do
        if [[ size -lt $N_NODES && " ${SKIP[@]} " =~ " ${algo} " ]]; then
            echo "Skipping algorithm $algo: is in SKIP and size=$size < N_NODES=$N_NODES"
            continue
        fi

        echo "Debugging -> Algo $algo, $N_NODES processes, $size array size"
        $RUN $RUNFLAGS -n $N_NODES $DEBUG_EXEC $size
    done
done

if [ $location != 'local' ]; then
    srun -n $N_NODES hostname > "$OUTPUT_DIR/$N_NODES.txt"
fi
