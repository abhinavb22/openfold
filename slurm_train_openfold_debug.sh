#!/bin/bash
#SBATCH -J of_train          # Job name
#SBATCH -o /work/10110/abhinav22/ls6/openfold/outerror/out.o%j       # Name of stdout output file
#SBATCH -e /work/10110/abhinav22/ls6/openfold/outerror/error.e%j       # Name of stderr error file
#SBATCH -p gpu-a100-dev          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH --ntasks-per-node=3              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 0-00:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=fail    # Send email at begin and end of job
#SBATCH -A ASC24025       # Project/Allocation name (req'd if you have more than 1)
#SBATCH --mail-user=abhinav22@tamu.edu

module purge

module load gcc/11.2.0
module load cuda/12.2
source /work/10110/abhinav22/ls6/src/miniconda/etc/profile.d/conda.sh
conda activate /work/10110/abhinav22/ls6/src/miniforge/envs/openfold_cuda12

#export CUDA_VISIBLE_DEVICES=0,1
#echo $CUDA_VISIBLE_DEVICES
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT=12355
./openfold_training_debug.sh
