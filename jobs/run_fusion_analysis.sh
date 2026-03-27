#!/bin/bash
#SBATCH --job-name=Fusion_Analysis
#SBATCH --output=logs/fusion_analysis_%j.out
#SBATCH --error=logs/fusion_analysis_%j.err
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
export DATA_ROOT=/home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split

echo ""
echo "========================================"
echo "Fusion Model Analysis (Per-class + Confusion Matrix)"
echo "========================================"
python compare_models.py \
    --swin_ckpt baselines/swin_baseline/checkpoints/best_swin.pth \
    --gavit_ckpt checkpoints/best_gavit_K9_spatial_knn_fusion.pth \
    --grouping spatial \
    --num_regions 9 \
    --knn_k 5 \
    --tag fusion

echo ""
echo "All analyses complete."