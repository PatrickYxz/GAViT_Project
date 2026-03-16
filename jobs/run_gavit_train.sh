#!/bin/bash
#SBATCH --job-name=GAViT_Train
#SBATCH --output=logs/gavit_train_%j.out
#SBATCH --error=logs/gavit_train_%j.err
#SBATCH --gpus=1
#SBATCH --constraint=gpu
#SBATCH --time=06:00:00

mkdir -p logs

module load Miniforge3
source activate gavit

echo "===== ENV INFO ====="
which python
python -c "import torch; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import torch_geometric; print('PyG:', torch_geometric.__version__)"
nvidia-smi
echo "===================="

cd /home/yang1004/GAViT_Project/
export DATA_ROOT=/home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split

python train_gavit.py
