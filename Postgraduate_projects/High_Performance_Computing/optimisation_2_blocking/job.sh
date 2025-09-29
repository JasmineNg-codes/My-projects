#!/bin/bash
#SBATCH -J heat_diffusion_job      # Job name
#SBATCH -A MPHIL-DIS-SL2-CPU       # Project account
#SBATCH -p icelake                 # Partition
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                # Number of tasks
#SBATCH --time=00:01:00            # Walltime limit
#SBATCH --mail-type=ALL            # Email notifications for job start/end
#SBATCH --no-requeue               # Prevent job from being requeued

# Load required modules
module purge
module load rhel8/default-icl
module load cmake
module load gcc

# Navigate to the code directory
cd "$SLURM_SUBMIT_DIR"

make clean
make 
make run
make valgrind-profile