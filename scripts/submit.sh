#!/bin/bash
#SBATCH -p <p_name>                  # Partition name
# #SBATCH -q <qos_name>              # Required Quality of Service (optional)
#SBATCH -N 16                        # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of taasks per node
#SBATCH --time=<requested_time>      # Time limit
#SBATCH --exclusive                  # Exclusive access to nodes
#SBATCH --account=<account_name>     # Account name

TIMESTAMP=$(date +"%Y_%m_%d___%H:%M:%S")

/home/spasqualoni/Swing_Test/run_test_suite.sh 16 $TIMESTAMP
/home/spasqualoni/Swing_Test/run_test_suite.sh 8 $TIMESTAMP
/home/spasqualoni/Swing_Test/run_test_suite.sh 4 $TIMESTAMP
/home/spasqualoni/Swing_Test/run_test_suite.sh 2 $TIMESTAMP
