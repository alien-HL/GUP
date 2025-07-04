CUDA_VISIBLE_DEVICES=0 python visualize.py \
  --features_dir /home/huanglai/SIMPLArgoverse/features/ \
  --mode val \
  --use_cuda \
  --model_path saved_models/20250406-210153_GUP_best.tar\
  --adv_cfg_path config.GUP_cfg \
  --visualizer GUP.av1_visualizer:Visualizer \
  --seq_id -1
  #--features_dir /home/huanglai/Argoverse/sample/vis/features/ \
