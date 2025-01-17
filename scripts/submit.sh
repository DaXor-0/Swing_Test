#!/bin/bash
#SBATCH -p <p_name>                  # Partition name
# #SBATCH -q <qos_name>              # Required Quality of Service (optional)
#SBATCH -N <n_nodes>                 # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of taasks per node
#SBATCH --time=<requested_time>      # Time limit
#SBATCH --exclusive                  # Exclusive access to nodes
#SBATCH --account=<account_name>     # Account name

TIMESTAMP=$(date +"%Y_%m_%d___%H:%M:%S")
LOCATION=leonardo # possible locations are leonardo, snellius and local
CUDA=no
OMPI_TEST=yes

$HOME/Swing_Test/scripts/run_test_suite.sh $SLURM_NNODES $TIMESTAMP $LOCATION $CUDA $OMPI_TEST
