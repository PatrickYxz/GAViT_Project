#!/bin/bash
#SBATCH --job-name=Edge_Vis
#SBATCH --output=logs/edge_vis_%j.out
#SBATCH --error=logs/edge_vis_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00

mkdir -p logs

module load Miniforge3
source activate
conda activate gavit

cd /home/yang1004/GAViT_Project/
export DATA_ROOT=/home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split

echo "===== Edge Comparison Visualization ====="
python visualize_edge_comparison.py \
    --ckpt_spatial checkpoints/best_gavit_K9_spatial_spatial.pth \
    --ckpt_knn checkpoints/best_gavit_K9_spatial_knn.pth \
    --classes airport stadium harbor dense_residential forest desert \
    --num_per_class 2

echo "Done."