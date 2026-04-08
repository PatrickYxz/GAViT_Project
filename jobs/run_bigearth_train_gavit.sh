#!/bin/bash
#SBATCH --job-name=BE_GAViT
#SBATCH --output=logs/bigearth_train_gavit_%j.out
#SBATCH --error=logs/bigearth_train_gavit_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

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
export BIGEARTH_ROOT=/home/yang1004/GAViT_Project/datasets/BigEarthNet-RGB_split

echo "===== BigEarthNet: GAViT v2 K=16 attentive_spatial + token_feedback ====="
python train_bigearth.py \
    --model gavit \
    --grouping attentive_spatial \
    --integration token_feedback \
    --num_regions 16 \
    --edge_type knn \
    --knn_k 5 \
    --epochs 30 \
    --resume \
    --start_epoch 25