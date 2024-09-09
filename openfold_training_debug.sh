#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
#export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1

DATA_DIR=/scratch/10110/abhinav22
TEMPLATE_MMCIF_DIR=/work/10110/abhinav22/ls6/database_2024_07_06/pdb_mmcif
CHECKPOINT_PATH=openfold/resources/params/params_model_1_multimer_v3.npz
CACHE_DIR=$DATA_DIR/cache_dir

python3 train_openfold.py $DATA_DIR/mmcif_dir/train $DATA_DIR/alignment_dir/train $TEMPLATE_MMCIF_DIR/mmcif_files $DATA_DIR/output_dir 2021-09-30 \
	  --config_preset model_1_multimer_v3 \
	  --template_release_dates_cache_path $CACHE_DIR/template_mmcif_cache.json \
    --seed 77843 \
    --obsolete_pdbs_file_path $TEMPLATE_MMCIF_DIR/obsolete.dat \
    --num_nodes 1 \
    --resume_from_jax_params $CHECKPOINT_PATH \
  	--resume_model_weights_only False \
  	--train_mmcif_data_cache_path $CACHE_DIR/train_mmcif_cache.json \
  	--val_mmcif_data_cache_path $CACHE_DIR/val_mmcif_cache.json \
  	--val_data_dir $DATA_DIR/mmcif_dir/val \
  	--val_alignment_dir $DATA_DIR/alignment_dir/val \
    --gpus 3 \
  	--train_epoch_len 5 \
  	--max_epochs 1 \
  	--checkpoint_every_epoch  \
  	--precision bf16-mixed \
  	--num_sanity_val_steps 0 \
  	--log_performance False \
  	--wandb \
  	--log_every_n_steps 1 \
  	--log_lr \
  	--wandb_project openfold_training \
  	--experiment_name full_train \
	 	--deepspeed_config_path ./deepspeed_config.json 
#  	--mpi_plugin 
 
  	
