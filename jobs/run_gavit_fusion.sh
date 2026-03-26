#!/bin/bash
#SBATCH --job-name=GAViT_Fusion
#SBATCH --output=logs/gavit_fusion_%j.out
#SBATCH --error=logs/gavit_fusion_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --time=12:00:00

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
# GAViT v2 (with backbone-graph fusion), 50 epochs
# ------------------------------------------------------------------
echo "===== GAViT K=9 spatial + fusion + 50 epochs ====="
python train_gavit.py \
    --grouping spatial \
    --num_regions 9 \
    --knn_k 5 \
    --gat_layers 2 \
    --gat_heads 4 \
    --edge_type knn \
    --epochs 50