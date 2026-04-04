#!/bin/bash
#SBATCH --job-name=GAViT_v2
#SBATCH --output=logs/gavit_v2_%j.out
#SBATCH --error=logs/gavit_v2_%j.err
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
# GAViT v2: AttentiveSpatialGrouping K=16 + token_feedback, 30 epochs
# ------------------------------------------------------------------
echo "===== GAViT v2: K=16 attentive_spatial + token_feedback ====="
python train_gavit.py \
    --grouping attentive_spatial \
    --num_regions 16 \
    --knn_k 5 \
    --gat_layers 2 \
    --gat_heads 4 \
    --edge_type knn \
    --integration token_feedback \
    --epochs 30