#!/bin/bash
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

torchrun --nproc_per_node=2 train_openfold.py $DATA_DIR/mmcif_dir/train $DATA_DIR/alignment_dir/train $TEMPLATE_MMCIF_DIR/mmcif_files $DATA_DIR/output_dir 2021-09-30 \
	--config_preset model_1_multimer_v3 \
	--template_release_dates_cache_path $CACHE_DIR/mmcif_cache.json \
    --seed 77843 \
    --obsolete_pdbs_file_path $TEMPLATE_MMCIF_DIR/obsolete.dat \
    --num_nodes 2 \
    --resume_from_jax_params $CHECKPOINT_PATH \
  	--resume_model_weights_only False \
  	--train_mmcif_data_cache_path $CACHE_DIR/train_mmcif_cache.json \
  	--val_mmcif_data_cache_path $CACHE_DIR/val_mmcif_cache.json \
  	--val_data_dir $DATA_DIR/mmcif_dir/val \
  	--val_alignment_dir $DATA_DIR/alignment_dir/val \
    --gpus 2 \
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


  	
