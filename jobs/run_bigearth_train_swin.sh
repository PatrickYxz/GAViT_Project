#!/bin/bash
#SBATCH --job-name=BE_Swin
#SBATCH --output=logs/bigearth_train_swin_%j.out
#SBATCH --error=logs/bigearth_train_swin_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=16:00:00

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

echo "===== BigEarthNet: Swin-T Baseline ====="
python train_bigearth.py \
    --model swin \
    --epochs 30 \
    --resume \
    --start_epoch 17