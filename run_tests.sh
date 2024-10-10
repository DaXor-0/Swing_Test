#!/bin/bash

# Paths
EXECUTABLE=./out
# RULES_FILE_PATH=./collective_rules.txt
ABS_PATH=/home/saverio/University/Tesi/test/collective_rules.txt

# Array of process counts and array sizes
PROCESSES=(
  2
  4
  8
  # 16
  # 32
  # 64
  # 128
  # 256
  # 512
  # 1024
  # 2048
  # 4096
  # 8192
  # 16384
)
ARRAY_SIZES=(
  10
  100
  1000
  10000
  100000
  # 1000000
  # 10000000
  # 100000000
  # 1000000000
  # 10000000000
)

# export "UCX_IB_SL=1"

# Run the tests for each number from 1 to 12
for algo in {1..12}; do
    # Update the collective_rules.txt using the C program
    ./update_collective_rules $algo
    export "OMPI_MCA_coll_tuned_use_dynamic_rules=1"
    # export "OMPI_MCA_coll_tuned_dynamic_rules_filename=${RULES_FILE_PATH}"
    export "OMPI_MCA_coll_tuned_dynamic_rules_filename=${ABS_PATH}"

    # Run the tests with different process counts and array sizes
    for proc in "${PROCESSES[@]}"; do
        for size in "${ARRAY_SIZES[@]}"; do
            if (( size < proc )); then
                echo "Skipping: array size $size <= number of processes $proc (Algo: $algo)"
                continue
            fi

            echo "Running with $proc processes and array size $size (Algo: $algo)"
            mpirun -np $proc $EXECUTABLE $size 50
        done
    done
done
