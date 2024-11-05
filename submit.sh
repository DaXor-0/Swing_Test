#!/bin/bash
#SBATCH -p <p_name>                  # Partition name
#SBATCH -N 16                        # Number of nodes
#SBATCH --ntasks-per-node=1          # Number of taasks per node
#SBATCH --time=01:30:00              # Time limit
#SBATCH --exclusive                  # Exclusive access to nodes
#SBATCH --account=<account_name>     # Account name

/home/spasqualoni/Swing_Test/run_tests.sh 16                # run the test suite on 16 node

scontrol update JobId=$SLURM_JOB_ID NumNodes=8
. slurm_job_${SLURM_JOB_ID}_resize.sh

/home/spasqualoni/Swing_Test/run_tests.sh 8                 # run the test suite on 8 node

scontrol update JobId=$SLURM_JOB_ID NumNodes=4
. slurm_job_${SLURM_JOB_ID}_resize.sh

/home/spasqualoni/Swing_Test/run_tests.sh 4                 # run the test suite on 4 node

scontrol update JobId=$SLURM_JOB_ID NumNodes=2
. slurm_job_${SLURM_JOB_ID}_resize.sh

/home/spasqualoni/Swing_Test/run_tests.sh 2                 # run the test suite on 2 node
