# Semantic-Segmentation-Cityscapes
This repository contains a preprocessing pipeline for the Cityscapes Dataset tailored for semantic segmentation tasks. It simplifies the original class space, resizes and normalizes images, and prepares train/val/test splits with processed masks.
# Folder Structure
Downlaod the Cityscapes dataset from  https://www.cityscapes-dataset.com/downloads/ , and makes sure your folder structure looks like following:

├── leftImg8bit/                
│   ├── train/ \
│   ├── val/ \
│   └── test/ \
├── gtFine/                   
│   ├── train/ \
│   ├── val/ \
│   └── test/ \
├── output/ \
│   ├── images/                 
│   ├── masks/                  
│   ├── images_processed/        
│   └── masks_processed/        
# Preprocessing Features
1.Class simplification (34 → 7 or fewer classes)
2.Per-pixel relabeling 
3.Image resizing to 256×256
4.RGB image normalization to [0, 1]
5.Nearest-neighbor resizing for masks
6.Saves stats per split
# Dependencies
uv venv \
source .venv/bin/activate \
uv pip install -r requirements.txt
# Run Preprocessing
python preprocess.py \
    --data_dir ./data/cityscapes \
    --output_dir ./output \
    --splits train val test \
    --log_level INFO
# Output Stats
After running, check files like:stats_train.json, stats_val.json. These include edge cases, pixel distributions, and total images processed.
# Sample Outputs
You’ll find visualizations and mask overlays in output/visualizations/. Some visualizations are as followings:
![samples_train](https://github.com/user-attachments/assets/473e3189-a608-46b9-9541-51f54de38ac2)

# Report
A detailed report covering decisions, visuals, and dataset insights is included in the report/ folder.
