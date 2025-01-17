#!/bin/bash

cleanup() {
    echo "Caught Ctrl+C! Stopping the script and killing all child processes..."
    pkill -P $$    # Kills all processes whose parent is the current script
    exit 1         # Exit the script with a non-zero status
}

trap cleanup SIGINT

N_NODES=$1
TIMESTAMP=$2

location='local'              # This refers to SAVERIO's laptop, if you want to run on your local machine, modify local or add a new one
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

ALGOS=(3 8 6 13)
# ALGOS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13)
SKIP=(4 5 6 9 10 11 12 13 16)

ARR_SIZES=(8 64 512 2048 16384) #131072 1048576 8388608 67108864)
TYPES=('int64' )
# TYPES=('int32' 'int64' 'float' 'double' 'char' 'int8' 'int16')
# NOTE: problems with char, int8, int16


OUTPUT_DIR="$RES_DIR/$location/$TIMESTAMP/"
DATA_DIR="$OUTPUT_DIR/data/"

get_iterations() {
    size=$1
    if [ $size -le 512 ]; then
        echo 15000
    elif [ $size -le 1048576 ]; then
        echo 1500
    elif [ $size -le 8388608 ]; then
        echo 100
    elif [ $size -le 67108864 ]; then
        echo 15
    else
        echo 5
    fi
}


run_test() {
    local size=$1
    local iter=$2
    local type=$3
    local algo=$4

    echo "Running -> $N_NODES processes, $size array size, $type datatype (Algo: $algo)"
    $RUN $RUNFLAGS -n $N_NODES $TEST_EXEC $size $iter $type $algo $DATA_DIR
}

mkdir -p "$RES_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

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

        iter=$(get_iterations $size)
        for type in "${TYPES[@]}"; do
            run_test $size $iter $type $algo
        done
    done
done

if [ $location != 'local' ]; then
    srun -n $N_NODES hostname > "$OUTPUT_DIR/$N_NODES.txt"
fi
