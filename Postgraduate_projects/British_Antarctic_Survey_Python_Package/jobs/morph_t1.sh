#!/bin/bash
#SBATCH -J morph_t1            # Job name
#SBATCH -A MPHIL-DIS-SL2-CPU   # Project account
#SBATCH -p icelake-himem       # Partition
#SBATCH --nodes=1              # One node (sufficient for your workload)
#SBATCH --cpus-per-task=4      # Adjust based on how parallel your Python is
#SBATCH --ntasks=1             # One task (no MPI)
#SBATCH --time=02:30:00        # Max wall time
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --output=morph_t1.out

# === Load environment ===
module purge
module load miniconda3

# Go to submission directory (very important!)
cd "$SLURM_SUBMIT_DIR"

# Activate your environment
source delphienv/bin/activate

# Optional: install your package (if setup.py is in jn492/)
pip install -e .

export PATH="$HOME/.local/bin:$PATH"

# Run your script
python scripts/morph_t1.py