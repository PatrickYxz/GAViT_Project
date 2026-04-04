#!/bin/bash
#SBATCH --job-name=BigEarth_Prep
#SBATCH --output=logs/bigearth_prepare_%j.out
#SBATCH --error=logs/bigearth_prepare_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

mkdir -p logs

module load Miniforge3
source activate
conda activate gavit

cd /home/yang1004/GAViT_Project/

# Adjust BIGEARTH_DIR to wherever you extracted BigEarthNet-RGB
BIGEARTH_DIR=/home/yang1004/datasets/BigEarthNet-RGB

echo "===== Preparing BigEarthNet-RGB splits ====="
python baselines/bigearth/prepare_bigearth.py \
    --data_dir  $BIGEARTH_DIR \
    --out_dir   datasets/BigEarthNet-RGB_split

echo "===== Done ====="