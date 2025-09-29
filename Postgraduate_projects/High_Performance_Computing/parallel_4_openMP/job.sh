#!/bin/bash
#SBATCH -J heat_diffusion_openmp       # Job name
#SBATCH -A MPHIL-DIS-SL2-CPU           # Project account
#SBATCH -p icelake                     # Partition
#SBATCH --nodes=1                      # One node
#SBATCH --cpus-per-task=76             # OpenMP threads
#SBATCH --ntasks=1                     # One MPI rank (OpenMP only)
#SBATCH --time=00:10:00                # Max wall time
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out

# === Load environment ===
module purge
module load rhel8/default-icl
module load gcc
module load cmake

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"

make clean
make
make run
make valgrind-profile
