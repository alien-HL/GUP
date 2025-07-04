echo "-- Processing val set..."
python data_argo/run_preprocess.py --mode val \
  --data_dir /home/huanglai/SIMPLArgoverse/Argoverse/forecasting/val/data/ \
  --save_dir /home/huanglai/SIMPLArgoverse/Argoverse/forecasting/features

echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
  --data_dir /home/huanglai/SIMPLArgoverse/Argoverse/forecasting/train/data/ \
  --save_dir /home/huanglai/SIMPLArgoverse/Argoverse/forecasting/features

echo "-- Processing test set..."
python data_argo/run_preprocess.py --mode test \
  --data_dir /home/huanglai/SIMPLArgoverse/Argoverse/forecasting/test/data/ \
  --save_dir /home/huanglai/SIMPLArgoverse/Argoverse/forecasting/features

