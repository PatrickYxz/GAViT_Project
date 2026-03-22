#!/bin/bash
#SBATCH --job-name=GAViT_Train
#SBATCH --output=logs/gavit_train_%j.out
#SBATCH --error=logs/gavit_train_%j.err
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
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
nvidia-smi
echo "===================="

cd /home/yang1004/GAViT_Project/
export DATA_ROOT=/home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split

# ------------------------------------------------------------------
# Experiment P2a: GAViT K=9 SpatialGrouping + 2-layer GAT
# (pairs with P1 spatial to isolate the GNN contribution)
# ------------------------------------------------------------------
echo "===== Experiment: GAViT K=9 spatial + 2-layer GAT ====="
python train_gavit.py --grouping spatial --num_regions 9 --knn_k 5 --gat_layers 2 --gat_heads 4

# ------------------------------------------------------------------
# Experiment P2b: GAViT K=9 KMeansGrouping + 2-layer GAT
# (full model with learned grouping)
# ------------------------------------------------------------------
echo "===== Experiment: GAViT K=9 kmeans + 2-layer GAT ====="
python train_gavit.py --grouping kmeans --num_regions 9 --knn_k 5 --gat_layers 2 --gat_heads 4
