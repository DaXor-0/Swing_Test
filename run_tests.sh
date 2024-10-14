#!/bin/bash

cleanup() {
    echo "Caught Ctrl+C! Stopping the script and killing all child processes..."
    pkill -P $$    # Kills all processes whose parent is the current script
    exit 1         # Exit the script with a non-zero status
}

trap cleanup SIGINT

# Setup for local tests or for leonardo tests
# location='local'
location='leonardo'

if [ $location == 'leonardo' ]; then
    export PATH=/leonardo/home/userexternal/spasqual/bin:$PATH
    export LD_LIBRARY_PATH=/leonardo/home/userexternal/spasqual/lib:$LD_LIBRARY_PATH
    export MANPATH=/leonardo/home/userexternal/spasqual/share/man:$MANPATH

    export "OMPI_MCA_coll_hcoll_enable=0"
    export "UCX_IB_SL=1"
    
    export CUDA_VISIBLE_DEVICES=""
    export OMPI_MCA_btl="^smcuda"
    export OMPI_MCA_mpi_cuda_support=0
    
    RUN=srun
    TEST_EXEC=/leonardo/home/userexternal/spasqual/Swing_Test/out
    RULE_UPDATER_EXEC=/leonardo/home/userexternal/spasqual/Swing_Test/update_collective_rules
    RULE_FILE_PATH=/leonardo/home/userexternal/spasqual/Swing_Test/collective_rules.txt
elif [ $location == 'local' ]; then
    export PATH=/opt/ompi_test/bin:$PATH
    export LD_LIBRARY_PATH=/opt/ompi_test/lib:$LD_LIBRARY_PATH
    export MANPATH=/opt/ompi_test/share/man:$MANPATH
    
    export "OMPI_MCA_coll_hcoll_enable=0"
    
    RUN=mpiexec
    TEST_EXEC=./out
    RULE_UPDATER_EXEC=./update_collective_rules
    RULE_FILE_PATH=/home/saverio/University/Tesi/test/collective_rules.txt
else
    echo "ERROR: location not correctly set up, aborting..."
    exit 1
fi


RES_DIR=./results/
TIMESTAMP=$(date +"%Y_%m_%d___%H:%M:%S")
OUTPUT_DIR="$RES_DIR/$TIMESTAMP/"

N_PROC=(2 4 8 16 32 64 128 256 512 1024 2048 4096 8192) # 16384)
ARR_SIZES=(8 64 512 2048 16384 131072 1048576 8388608 67108864 536870912)
TYPE=int


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
    local algo=$4

    echo "Running with $n processes and array size $size (Algo: $algo)"
    $RUN -n $n $TEST_EXEC $size $iter $TYPE $RULE_FILE_PATH $OUTPUT_DIR
}


mkdir -p "$RES_DIR"
mkdir -p "$OUTPUT_DIR"


# Test mpi baseline, loop through:
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
export "OMPI_MCA_coll_tuned_use_dynamic_rules=0"
for n in "${N_PROC[@]}"; do
    for size in "${ARR_SIZES[@]}"; do
        if (( size < n )); then
            echo "Skipping: array size $size <= number of processes $n (BASELINE)"
            continue
        fi
        iter=$(get_iterations $size)
        run_test $n $size $iter "BASELINE"
    done
done


# Test custom algorithms here, loop through:
# - algorithms:
#     - 8 swing latency
#     - 9 swing bandwidt memcp
#     - 10 swing bandwidt datatype
#     - 11 swing bandwidt datatype + memcp
#     - 12 swing bandwidt segmented
# - number of mpi processes
# - size of the array in number of elements (of type = TYPE) 
export "OMPI_MCA_coll_tuned_use_dynamic_rules=1"
for algo in $(seq 1 12); do
    $RULE_UPDATER_EXEC $RULE_FILE_PATH $algo
    export "OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_PATH}"

    for n in "${N_PROC[@]}"; do
        for size in "${ARR_SIZES[@]}"; do
            if (( size < n )); then
                echo "Skipping: array size $size <= number of processes $n (Algo: $algo)"
                continue
            fi

            iter=$(get_iterations $size)
            run_test $n $size $iter $algo
        done
    done
done
