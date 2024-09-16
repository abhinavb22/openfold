#!/bin/bash
#SBATCH --job-name=prediction_0   # job name
#SBATCH --time=0-12:00:00   	   # max job run time dd-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20          # CPUs (threads) per command
#SBATCH --mem=100G                  # total memory per node
#SBATCH --gres=gpu:t4:1           # request 1 A100 GPU
#SBATCH --partition=gpu
#SBATCH --account=146392998287
#SBATCH --output=/scratch/user/u.as139299/outerror/out_.%j.%x	# save stdout to file
#SBATCH --error=/scratch/user/u.as139299/outerror/error_.%j.%x        # save stderr to file

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --mail-type=FAIL              #Send email on all job events
#SBATCH --mail-user=abhinav22@tamu.edu    #Send all emails to email_address


module purge
module load GCC/10.3.0
module load CUDA/12.1.0
module load Anaconda3/2024.02-1

source activate openfold_cuda12

#srun --ntasks 2 ./openfold_training_debug.sh
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
#export MASTER_PORT=12355
#export WORLD_SIZE=2  # Since you are using 2 GPUs
#export RANK=0 

DATA_DIR=/scratch/group/alphafold_classifier/openfold_files/openfold_training/oligomers
TEMPLATE_MMCIF_DIR=/scratch/user/u.as139299/database/pdb_mmcif
CHECKPOINT_PATH=openfold/resources/params/params_model_1_multimer_v3.npz
CACHE_DIR=$DATA_DIR/cache_dir

nvidia-smi

# Generate a unique port number based on the job ID
export MASTER_PORT=$(shuf -i 29400-65000 -n 1)

# Get the hostname of the first node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train_openfold.py $DATA_DIR/mmcif_dir/train $DATA_DIR/alignment_dir/train $TEMPLATE_MMCIF_DIR/mmcif_files /scratch/user/u.as139299/output_dir 2021-09-30 \
	--config_preset model_1_multimer_v3 \
	--template_release_dates_cache_path $CACHE_DIR/mmcif_cache.json \
    --seed 77843 \
    --obsolete_pdbs_file_path $TEMPLATE_MMCIF_DIR/obsolete.dat \
    --num_nodes 1 \
    --resume_from_jax_params $CHECKPOINT_PATH \
  	--resume_model_weights_only False \
  	--train_mmcif_data_cache_path $CACHE_DIR/train_mmcif_cache.json \
  	--val_mmcif_data_cache_path $CACHE_DIR/val_mmcif_cache.json \
  	--val_data_dir $DATA_DIR/mmcif_dir/val \
  	--val_alignment_dir $DATA_DIR/alignment_dir/val \
    --gpus 1 \
  	--train_epoch_len 1000 \
  	--max_epochs 1 \
  	--checkpoint_every_epoch  \
  	--precision 32 \
  	--num_sanity_val_steps 0 \
  	--log_performance False \
  	--log_every_n_steps 1 \
 	--log_lr \
  	--wandb \
  	--wandb_project openfold_training \
  	--experiment_name deepspeed_mock_faster \
  	--deepspeed_config_path ./deepspeed_config.json 
#  	--mpi_plugin 