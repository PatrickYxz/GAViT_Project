
#!/bin/bash
#SBATCH --job-name=BigEarth_DL
#SBATCH --output=logs/bigearth_download_%j.out
#SBATCH --error=logs/bigearth_download_%j.err
#SBATCH --time=12:00:00

mkdir -p logs

cd /projects/gavitdata/

echo "===== Downloading BigEarthNet-S2-v1.0 from Zenodo ====="
date
wget -c "https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz?download=1" \
    -O BigEarthNet-S2-v1.0.tar.gz
echo "Download complete."
date

echo "===== Selective extraction: RGB bands + labels JSON ====="
tar -xzf BigEarthNet-S2-v1.0.tar.gz \
    --wildcards '*_B02.tif' '*_B03.tif' '*_B04.tif' '*_labels_metadata.json'
echo "Extraction complete."
date

echo "===== Removing tar.gz to free space ====="
rm BigEarthNet-S2-v1.0.tar.gz

echo "===== Verifying ====="
du -sh /projects/gavitdata/
ls -d /projects/gavitdata/BigEarthNet-v1.0/*/ 2>/dev/null | wc -l
echo "===== Done ====="
date
