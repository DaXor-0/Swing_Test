#!/bin/bash

cleanup() {
    echo "Caught Ctrl+C! Stopping the script and killing all child processes..."
    pkill -P $$    # Kills all processes whose parent is the current script
    exit 1         # Exit the script with a non-zero status
}

trap cleanup SIGINT

# Setup for local tests or for leonardo tests
location='local'
# location='leonardo'
debug=yes
#debug=no


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
    RULE_UPDATER_SCRIPT=/leonardo/home/userexternal/spasqual/Swing_Test/change_collective.sh
    
    ALGOS=(0 1 2 3 4 5 6 7 8 9 10 11 12)
    N_PROC=(2 4 8 16 32 64 128 256 512 1024 2048) # 4096 8192 16384)
    ARR_SIZES=(8 64 512 2048 16384 131072 1048576 8388608 67108864) # 536870912)
    TYPES=('int32' 'int64' 'float' 'double') # 'char' 'int8' 'int16')
elif [ $location == 'local' ]; then
    # sets PATH, LD_LIBRARY_PATH and MANPATH
    source ~/use_ompi.sh

    RUN=mpiexec
    RES_DIR=./local_results/
    TEST_EXEC=./out
    DEBUG_EXEC=./debug
    RULE_UPDATER_SCRIPT=./change_collective.sh

    ALGOS=(0 8 9 10 11 12)
    N_PROC=(8)
    ARR_SIZES=(16384)
    TYPES=('int32' 'int64')
    # no problems with int, int32, int64, float, double
    # problems with char, int8, int16
else
    echo "ERROR: location not correctly set up, aborting..."
    exit 1
fi


export OMPI_MCA_coll_hcoll_enable=0
export OMPI_MCA_coll_tuned_use_dynamic_rules=1
chmod +x $RULE_UPDATER_SCRIPT
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
        echo 10
    else
        echo 5
    fi
}


run_test() {
    local n=$1
    local size=$2
    local iter=$3
    local type=$4
    local algo=$5
    
    echo "Running -> $n processes, $size array size, $type datatype (Algo: $algo)"
    $RUN -n $n $TEST_EXEC $size $iter $type $algo $OUTPUT_DIR
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
    $RULE_UPDATER_SCRIPT $location $algo

    for n in "${N_PROC[@]}"; do
        for size in "${ARR_SIZES[@]}"; do
            if (( size < n )); then
                echo "Skipping -> $n processes, $size array size (Algo: $algo)"
                continue
            fi

            if [ $debug == 'yes' ]; then
                echo "Debugging -> Algo $algo, $n processes, $size array size"
                $RUN -n $n $DEBUG_EXEC $size
            else
                iter=$(get_iterations $size)
                for type in "${TYPES[@]}"; do
                    run_test $n $size $iter $type $algo
                done
            fi
        done
    done
done
