#!/bin/bash
#SBATCH --job-name=GAViT_v2_Test
#SBATCH --output=logs/gavit_v2_test_%j.out
#SBATCH --error=logs/gavit_v2_test_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00

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

echo "===== GAViT v2 Test: K=16 attentive_spatial + token_feedback ====="
python test_gavit.py \
    --ckpt checkpoints/best_gavit_K16_attentive_spatial_knn_token_feedback.pth \
    --grouping attentive_spatial \
    --integration token_feedback \
    --num_regions 16 \
    --edge_type knn \
    --knn_k 5