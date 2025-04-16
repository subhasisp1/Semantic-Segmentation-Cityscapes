# Semantic-Segmentation-Cityscapes
This repository contains a preprocessing pipeline for the Cityscapes Dataset tailored for semantic segmentation tasks. It simplifies the original class space, resizes and normalizes images, and prepares train/val/test splits with processed masks.
#Folder Structure
├── leftImg8bit/                # Raw input RGB images
│   ├── train/
│   ├── val/
│   └── test/
├── gtFine/                     # Raw input ground truth masks
│   ├── train/
│   ├── val/
│   └── test/
├── output/
│   ├── images/                 # RGB images (original size)
│   ├── masks/                  # Remapped masks (original size)
│   ├── images_processed/       # Resized + normalized images (.npy)
│   └── masks_processed/        # Resized masks (.png)
#Preprocessing Features
1.Class simplification (34 → 7 or fewer classes)
2.Per-pixel relabeling 
3.Image resizing to 256×256
4.RGB image normalization to [0, 1]
5.Nearest-neighbor resizing for masks
6.Saves stats per split
#Dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
#Run Preprocessing
python preprocess.py \
    --data_dir ./data/cityscapes \
    --output_dir ./output \
    --splits train val test \
    --log_level INFO
#Output Stats
After running, check files like:stats_train.json, stats_val.json. These include edge cases, pixel distributions, and total images processed.
#Sample Outputs
You’ll find visualizations and mask overlays in output/visualizations/.
#Report
A detailed report covering decisions, visuals, and dataset insights is included in the report/ folder.
