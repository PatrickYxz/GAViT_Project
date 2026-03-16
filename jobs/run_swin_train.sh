#!/bin/bash
#SBATCH --job-name=Swin_Train
#SBATCH --output=logs/swin_train_%j.out
#SBATCH --error=logs/swin_train_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --time=04:00:00

mkdir -p logs

module load Miniforge3
source activate
conda activate gavit

echo "===== ENV INFO ====="
which python
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "===================="

cd /home/yang1004/GAViT_Project/baselines/swin_baseline/

export DATA_ROOT=/home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split
python train_swin_baseline.py
