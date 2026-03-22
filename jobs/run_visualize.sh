#!/bin/bash
#SBATCH --job-name=GAViT_Vis
#SBATCH --output=logs/visualize_%j.out
#SBATCH --error=logs/visualize_%j.err
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
nvidia-smi
echo "===================="

cd /home/yang1004/GAViT_Project/
export DATA_ROOT=/home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split

# ------------------------------------------------------------------
# Visualize GAViT graph attention on diverse scene classes
# ------------------------------------------------------------------
echo "===== Graph Visualization: diverse classes ====="
python visualize_graph.py \
    --checkpoint checkpoints/best_gavit_K9_spatial.pth \
    --grouping spatial \
    --num_regions 9 \
    --classes airport stadium harbor dense_residential forest desert \
    --num_per_class 2

echo "Done."
