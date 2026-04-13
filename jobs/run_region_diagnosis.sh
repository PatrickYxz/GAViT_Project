#!/bin/bash
#SBATCH --job-name=RegionDiag
#SBATCH --output=logs/region_diagnosis_%j.out
#SBATCH --error=logs/region_diagnosis_%j.err
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
# Region diagnosis: visualize 4x4 regions + graph for key classes
# ------------------------------------------------------------------
echo "===== Region Diagnosis Visualization ====="
python visualize_region_diagnosis.py \
    --checkpoint checkpoints/best_gavit_K16_attentive_spatial_knn_token_feedback.pth \
    --num_regions 16 \
    --grouping attentive_spatial \
    --edge_type knn \
    --integration token_feedback \
    --classes airport bridge church stadium harbor forest \
    --num_per_class 2

echo "Done."