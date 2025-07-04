echo "-- Processing vis set..."
python data_argo/run_preprocess.py --mode val \
  --data_dir /home/huanglai/Argoverse/sample/vis/val/data/ \
  --save_dir /home/huanglai/Argoverse/sample/vis/features/

echo "-- Processing train set..."
python data_argo/run_preprocess.py --mode train \
  --data_dir /home/huanglai/Argoverse/sample/vis/train/data/ \
  --save_dir /home/huanglai/Argoverse/sample/vis/features/

