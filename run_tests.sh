#e/bin/bash

cleanup() {
    echo "Caught Ctrl+C! Stopping the script and killing all child processes..."
    pkill -P $$    # Kills all processes whose parent is the current script
    exit 1         # Exit the script with a non-zero status
}

trap cleanup SIGINT

# Paths
TEST_EXEC=./out

RULE_FILE_PATH=./collective_rules.txt
RULE_FILE_ABS_PATH=/home/saverio/University/Tesi/test/collective_rules.txt
RULE_UPDATER_EXEC=./update_collective_rules

RES_DIR=./results/
TIMESTAMP=$(date +"%Y_%m_%d___%H:%M:%S")
OUTPUT_DIR="$RES_DIR/$TIMESTAMP/"

# Array of process counts and array sizes
N_PROC=(
  2
  4
  8
  16
  32
  64
  128
  256
  512
  1024
  2048
  4096
  8192
  16384
)

ARR_SIZES=(
  1
  8
  64
  512
  2048
  16384
  131072
  1048576
  8388608
  67108864
)


type=int



# Check if the directory exists
if [ -d "$RES_DIR" ]; then
    echo "Directory $RES_DIR exists."
else
    echo "Directory $RES_DIR does not exist. Creating it..."
    mkdir -p "$RES_DIR"  # Create the directory if it doesn't exist
fi

mkdir -p "$OUTPUT_DIR"

export "UCX_IB_SL=1"

# Run the tests with different process counts and array sizes
export "OMPI_MCA_coll_tuned_use_dynamic_rules=0"
for n in "${N_PROC[@]}"; do
    for size in "${ARR_SIZES[@]}"; do
        iter=0
        if [ $size -le 512 ]; then
            iter=10000
        elif [ $size -le 1048576 ]; then
            iter=1000
        elif [ $size -le 8388608 ]; then
            iter=100
        elif [ $size -le 67108864 ]; then
            iter=10
        else
            iter=4
        fi
        if (( size < n )); then
            echo "Skipping: array size $size <= number of processes $n (BASELINE)"
            continue
        fi

        echo "Running with $n processes and array size $size (BASELINE)"
        mpirun -np $n $TEST_EXEC $size $iter $type $OUTPUT_DIR
    done
done

# Run the tests on specific algorithms
export "OMPI_MCA_coll_tuned_use_dynamic_rules=1"
for algo in {8..12}; do
    $RULE_UPDATER_EXEC $RULE_FILE_PATH $algo
    export "OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULE_FILE_ABS_PATH}"

    # Run the tests with different process counts and array sizes
    for n in "${N_PROC[@]}"; do
        for size in "${ARR_SIZES[@]}"; do
            iter=0
            if [ $size -le 512 ]; then
                iter=10000
            elif [ $size -le 1048576 ]; then
                iter=1000
            elif [ $size -le 8388608 ]; then
                iter=100
            elif [ $size -le 67108864 ]; then
                iter=10
            else
                iter=4
            fi
            if (( size < n )); then
                echo "Skipping: array size $size <= number of processes $n (Algo: {$algo})"
                continue
            fi

            echo "Running with $n processes and array size $size (Algo: {$algo})"
            mpirun -np $n $TEST_EXEC $size $iter $type $OUTPUT_DIR
        done
    done
done
