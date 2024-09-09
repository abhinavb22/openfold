#!/bin/bash
#SBATCH -J openfold_inference           # Job name
#SBATCH -o /work/10110/abhinav22/ls6/openfold/outerror/out.o%j       # Name of stdout output file
#SBATCH -e /work/10110/abhinav22/ls6/openfold/outerror/error.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-small          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 02:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=fail    # Send email at begin and end of job
#SBATCH -A ASC24025       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=abhinav22@tamu.edu

module purge

module load gcc/11.2.0
module load cuda/12.2
source /work/10110/abhinav22/ls6/src/miniconda/etc/profile.d/conda.sh
conda activate /work/10110/abhinav22/ls6/src/miniforge/envs/openfold_cuda12

./run_openfold.sh
