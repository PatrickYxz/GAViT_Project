#!/bin/bash
#SBATCH --job-name=BigEarth_Test
#SBATCH --output=logs/bigearth_test_%j.out
#SBATCH --error=logs/bigearth_test_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --time=01:00:00

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
export BIGEARTH_ROOT=/home/yang1004/GAViT_Project/datasets/BigEarthNet-RGB_split

# ------------------------------------------------------------------
# Swin-T Baseline
# ------------------------------------------------------------------
echo "===== BigEarthNet Test: Swin-T Baseline ====="
python test_bigearth.py \
    --model swin \
    --ckpt  checkpoints/best_bigearth_swin.pth

# ------------------------------------------------------------------
# GAViT v2
# ------------------------------------------------------------------
echo "===== BigEarthNet Test: GAViT v2 ====="
python test_bigearth.py \
    --model gavit \
    --ckpt  checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_token_feedback.pth \
    --num_regions 16 \
    --grouping attentive_spatial \
    --integration token_feedback