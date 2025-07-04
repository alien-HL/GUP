CUDA_VISIBLE_DEVICES=0 python evaluation.py \
  --features_dir /home/huanglai/SIMPLArgoverse/features/ \
  --train_batch_size 16 \
  --val_batch_size 128 \
  --use_cuda \
  --adv_cfg_path config.GUP_cfg \
  --model_path saved_models/20250406-210153_GUP_best.tar