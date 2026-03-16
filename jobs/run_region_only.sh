#!/bin/bash
#SBATCH --job-name=RegionOnly
#SBATCH --output=logs/region_only_%j.out
#SBATCH --error=logs/region_only_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --time=04:00:00

mkdir -p logs

module load Miniforge3
source activate gavit

echo "===== ENV INFO ====="
which python
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "===================="

cd /home/yang1004/GAViT_Project/
export DATA_ROOT=/home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split

# ------------------------------------------------------------------
# Experiment 2a: spatial grouping (deterministic, run once)
# Make sure GROUPING = "spatial" in train_region_only.py
# ------------------------------------------------------------------
echo "===== Experiment: Swin + SpatialGrouping K=9 ====="
python train_region_only.py

# ------------------------------------------------------------------
# Experiment 2b: kmeans grouping (has randomness, run 3 times)
# Change GROUPING = "kmeans" in train_region_only.py before submitting
# ------------------------------------------------------------------
# python train_region_only.py
