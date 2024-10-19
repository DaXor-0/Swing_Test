#!/bin/bash

cleanup() {
    echo "Caught Ctrl+C! Stopping the script and killing all child processes..."
    pkill -P $$    # Kills all processes whose parent is the current script
    exit 1         # Exit the script with a non-zero status
}

trap cleanup SIGINT

nnodes=$1

# location='local'
location='leonardo'

# debug=yes
debug=no

ALGOS=(0 1 2 3 4 5 6 7 8 9 10 11 12)
ARR_SIZES=(8 64 512 2048 16384 131072 1048576 8388608 67108864) # 536870912)
TYPES=('int32' 'int64' 'float' 'double') # 'char' 'int8' 'int16')
# NOTE: problems with char, int8, int16

# ALGOS=(0 8 9 10 11 12)
# ARR_SIZES=(8 64 512)
# TYPES=('int32' 'int64')

if [ $location == 'leonardo' ]; then
    export PATH=/leonardo/home/userexternal/spasqual/bin:$PATH
    export LD_LIBRARY_PATH=/leonardo/home/userexternal/spasqual/lib:$LD_LIBRARY_PATH
    export MANPATH=/leonardo/home/userexternal/spasqual/share/man:$MANPATH

    export UCX_IB_SL=1
    export CUDA_VISIBLE_DEVICES=""
    export OMPI_MCA_btl="^smcuda"
    export OMPI_MCA_mpi_cuda_support=0

    RUN=srun
    RES_DIR=./results/
    TEST_EXEC=/leonardo/home/userexternal/spasqual/Swing_Test/out
    DEBUG_EXEC=/leonardo/home/userexternal/spasqual/Swing_Test/debug
    RULE_UPDATER_EXEC=/leonardo/home/userexternal/spasqual/Swing_Test/update_collective_rules
    RULE_FILE_PATH=/leonardo/home/userexternal/spasqual/Swing_Test/collective_rules.txt
elif [ $location == 'local' ]; then
    # sets PATH, LD_LIBRARY_PATH and MANPATH
    source ~/use_ompi.sh

    RUN=mpiexec
    RES_DIR=./local_results/
    TEST_EXEC=./out
    DEBUG_EXEC=./debug
    RULE_UPDATER_EXEC=./update_collective_rules
    RULE_FILE_PATH=/home/saverio/University/Tesi/test/collective_rules.txt
else
    echo "ERROR: location not correctly set up, aborting..."
    exit 1
fi



export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
TIMESTAMP=$(date +"%Y_%m_%d___%H:%M:%S")
OUTPUT_DIR="$RES_DIR/$TIMESTAMP/"

get_iterations() {
    size=$1
    if [ $size -le 512 ]; then
        echo 10000
    elif [ $size -le 1048576 ]; then
        echo 1000
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
    
    # BUG: nnodes from here can create problems with the scope
    echo "Running -> $nnodes processes, $size array size, $type datatype (Algo: $algo)"
    $RUN -n $nnodes $TEST_EXEC $size $iter $type $algo $OUTPUT_DIR
}

if [ $debug == 'no' ]; then
    mkdir -p "$RES_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

# Test custom algorithms here, loop through:
# - algorithms:
#     - 8 swing latency
#     - 9 swing bandwidt memcp
#     - 10 swing bandwidt datatype
#     - 11 swing bandwidt datatype + memcp
#     - 12 swing bandwidt segmented
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
for algo in ${ALGOS[@]}; do   
    $RULE_UPDATER_EXEC $RULE_FILE_PATH $algo
    export OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_PATH}
    for size in "${ARR_SIZES[@]}"; do
        if (( size < nnodes )); then
            echo "Skipping -> $nnodes processes, $size array size (Algo: $algo)"
            continue
        fi

        if [ $debug == 'yes' ]; then
            echo "Debugging -> Algo $algo, $nnodes processes, $size array size"
            $RUN -n $nnodes $DEBUG_EXEC $size
        else
            iter=$(get_iterations $size)
            for type in "${TYPES[@]}"; do
                run_test $size $iter $type $algo
            done
        fi
    done
done
