#export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
#export SINGULARITYENV_TF_FORCE_UNIFIED_MEMORY=1
#export SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=4.0
#export CUDA_VISIBLE_DEVICES=0,1
output_dir=/work/10110/abhinav22/ls6/openfold_test/inference/result
fasta_dir=/work/10110/abhinav22/ls6/openfold_test/training/fasta_dir/val
download_dir=/work/10110/abhinav22/ls6/database_2024_07_06
precomputed_msas_dir=/work/10110/abhinav22/ls6/openfold_test/training/alignments_dir/val
openfold_checkpoint_path=/work/10110/abhinav22/ls6/openfold_test/training/output_dir/lightning_logs/1hc3ngjm/checkpoints/164-825.ckpt

python3 run_pretrained_openfold.py \
    $fasta_dir \
    $download_dir/pdb_mmcif/mmcif_files \
    --use_precomputed_alignments $precomputed_msas_dir \
    --config_preset "model_1_multimer_v3" \
    --openfold_checkpoint_path $openfold_checkpoint_path \
    --model_device "cuda:0" \
    --data_random_seed 77843 \
    --skip_relaxation \
    --save_output  \
    --output_dir $output_dir \
    --use_deepspeed_evoformer_attention 
