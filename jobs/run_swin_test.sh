#!/bin/bash
#SBATCH --job-name=Swin_Test
#SBATCH --output=logs/swin_test_%j.out
#SBATCH --error=logs/swin_test_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --time=00:30:00

mkdir -p logs

module load Miniforge3
source activate gavit

echo "===== ENV INFO ====="
which python
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
nvidia-smi
echo "===================="

cd /home/yang1004/GAViT_Project/baselines/swin_baseline/

python test_swin.py --ckpt checkpoints/best_swin.pth \
                    --data_root /home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split
