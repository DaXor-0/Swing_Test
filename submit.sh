#!/bin/bash
#SBATCH -p <p_name>                  # Partition name
#SBATCH -N 16                        # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of taasks per node
#SBATCH --time=01:30:00              # Time limit
#SBATCH --exclusive                  # Exclusive access to nodes
#SBATCH --account=<account_name>     # Account name

TIMESTAMP=$(date +"%Y_%m_%d___%H:%M:%S")

/home/spasqualoni/Swing_Test/run_tests.sh 16 $TIMESTAMP
/home/spasqualoni/Swing_Test/run_tests.sh 8 $TIMESTAMP
/home/spasqualoni/Swing_Test/run_tests.sh 4 $TIMESTAMP
/home/spasqualoni/Swing_Test/run_tests.sh 2 $TIMESTAMP
