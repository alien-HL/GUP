CUDA_VISIBLE_DEVICES=0 python train.py \
  --features_dir /home/huanglai/SIMPLArgoverse/features/ \
  --train_batch_size 8 \
  --val_batch_size 8 \
  --val_interval 2 \
  --train_epoches 50 \
  --data_aug \
  --use_cuda \
  --logger_writer \
  --adv_cfg_path config.GUP_cfg \
  #--resume \
  #--model_path /home/huanglai/下载/DGFNet-main/saved_models/20241228-195722_DGFNet_ckpt_epoch10.tar


