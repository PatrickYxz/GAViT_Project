#!/bin/bash
#SBATCH --job-name=Edge_Ablation
#SBATCH --output=logs/edge_ablation_%j.out
#SBATCH --error=logs/edge_ablation_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --time=08:00:00

mkdir -p logs

module load Miniforge3
source activate
conda activate gavit

echo "===== ENV INFO ====="
which python
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "===================="

cd /home/yang1004/GAViT_Project/
export DATA_ROOT=/home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split

# ------------------------------------------------------------------
# Edge Ablation: spatial adjacency vs cosine kNN vs hybrid
# All use: K=9 SpatialGrouping + 2-layer GAT (same as best P2a config)
# ------------------------------------------------------------------

echo "===== Edge Type: spatial adjacency ====="
python train_gavit.py --grouping spatial --num_regions 9 --gat_layers 2 --gat_heads 4 --edge_type spatial

echo "===== Edge Type: cosine kNN (baseline, for reference) ====="
python train_gavit.py --grouping spatial --num_regions 9 --gat_layers 2 --gat_heads 4 --edge_type knn

echo "===== Edge Type: hybrid (spatial + kNN) ====="
python train_gavit.py --grouping spatial --num_regions 9 --gat_layers 2 --gat_heads 4 --edge_type hybrid

echo "Done. All three edge ablation experiments completed."
