#!/bin/bash
#SBATCH -J heat_diff_tau
#SBATCH -A MPHIL-DIS-SL2-CPU
#SBATCH -p icelake
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out


# === Load modules ===
module purge
module load rhel8/default-icl
module load gcc
module load cmake
module load intel-oneapi-mpi/2021.6.0/intel

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"

# === Regular MPI build and run ===
make mpi_clean
make mpi
make mpi_run PROCS=16
