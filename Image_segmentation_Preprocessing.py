#!/usr/bin/env python3
"""
Cityscapes Dataset Preprocessing Script
Creates segmentation masks from Cityscapes dataset annotations
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cityscapes class mappings
CITYSCAPES_ID_TO_NAME = {
   0: 'unlabeled',
    1: 'ego vehicle',
    2: 'rectification border',
    3: 'out of roi',
    4: 'static',
    5: 'dynamic',
    6: 'ground',
    7: 'road',
    8: 'sidewalk',
    9: 'parking',
    10: 'rail track',
    11: 'building',
    12: 'wall',
    13: 'fence',
    14: 'guard rail',
    15: 'bridge',
    16: 'tunnel',
    17: 'pole',
    18: 'polegroup',
    19: 'traffic light',
    20: 'traffic sign',
    21: 'vegetation',
    22: 'terrain',
    23: 'sky',
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    29: 'caravan',
    30: 'trailer',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle',
    -1: 'license plate'
}

# Simplified class mapping (34 classes → 7 classes)
NEW_MAPPING = {
     0: 0,  # unlabeled -> background
    1: 0,  # ego vehicle -> background
    2: 0,  # rectification border -> background
    3: 0,  # out of roi -> background
    4: 0,  # static -> background
    5: 0,  # dynamic -> background
    6: 0,  # ground -> background
    7: 1,  # road -> road
    8: 0,  # sidewalk -> background
    9: 0,  # parking -> background
    10: 0, # rail track -> background
    11: 2, # building -> building
    12: 0, # wall -> background
    13: 0, # fence -> background
    14: 0, # guard rail -> background
    15: 0, # bridge -> background
    16: 0, # tunnel -> background
    17: 0, # pole -> background
    18: 0, # polegroup -> background
    19: 0, # traffic light -> background
    20: 0, # traffic sign -> background
    21: 5, # vegetation -> vegetation
    22: 0, # terrain -> background
    23: 6, # sky -> sky
    24: 4, # person -> person
    25: 4, # rider -> person
    26: 3, # car -> vehicle
    27: 3, # truck -> vehicle
    28: 3, # bus -> vehicle
    29: 3, # caravan -> vehicle
    30: 3, # trailer -> vehicle
    31: 3, # train -> vehicle
    32: 3, # motorcycle -> vehicle
    33: 3, # bicycle -> vehicle
    -1: 0, # license plate -> background
}

NEW_CLASS_NAMES = {
    0: 'background',
    1: 'road',
    2: 'building',
    3: 'vehicle',
    4: 'person',
    5: 'vegetation',
    6: 'sky'
}

def setup_directories(output_dir: str) -> None:
    """Create output directory structure"""
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks/test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    logger.info(f"Created directory structure in {output_dir}")

def process_image(
    img_path: str,
    gt_path: str,
    output_img_path: str,
    output_mask_path: str,
    stats: Dict
) -> bool:
    """Process single image and generate segmentation mask"""
    try:
        img = Image.open(img_path)
        gt = np.array(Image.open(gt_path))
        
        new_mask = np.zeros_like(gt, dtype=np.uint8)
        
        for old_id, new_id in NEW_MAPPING.items():
            new_mask[gt == old_id] = new_id
            
        unknown_pixels = np.sum(~np.isin(gt, list(NEW_MAPPING.keys())))
        if unknown_pixels > 0:
            stats['edge_cases']['unknown_class_ids'] += 1
            new_mask[~np.isin(gt, list(NEW_MAPPING.keys()))] = 0
            
        for class_id in range(len(NEW_CLASS_NAMES)):
            pixel_count = np.sum(new_mask == class_id)
            stats['class_pixels'][class_id] += pixel_count
            
        img.save(output_img_path)
        Image.fromarray(new_mask).save(output_mask_path)
        return True
        
    except Exception as e:
        logger.error(f"Error processing {img_path}: {str(e)}")
        stats['edge_cases']['processing_error'] += 1
        return False

def process_split(
    data_dir: str,
    output_dir: str,
    split: str = 'train',
    sample_size: Optional[int] = None
) -> Dict:
    """Process a dataset split (train/val/test)"""
    stats = {
        'edge_cases': defaultdict(int),
        'class_pixels': defaultdict(int),
        'total_images': 0
    }
    
    img_dir = os.path.join(data_dir, f'leftImg8bit/{split}')
    gt_dir = os.path.join(data_dir, f'gtFine/{split}')
    
    if not os.path.exists(img_dir) or not os.path.exists(gt_dir):
        logger.error(f"Missing directories for split {split}")
        return stats
        
    cities = os.listdir(img_dir)
    logger.info(f"Processing {split} split with {len(cities)} cities")
    
    count = 0
    for city in cities:
        city_img_dir = os.path.join(img_dir, city)
        city_gt_dir = os.path.join(gt_dir, city)
        
        img_files = sorted([f for f in os.listdir(city_img_dir) if f.endswith('leftImg8bit.png')])
        
        for img_file in tqdm(img_files, desc=f"Processing {city}"):
            if sample_size is not None and count >= sample_size:
                break
                
            base_name = img_file.replace('_leftImg8bit.png', '')
            gt_file = f"{base_name}_gtFine_labelIds.png"
            gt_path = os.path.join(city_gt_dir, gt_file)
            
            if not os.path.exists(gt_path):
                logger.warning(f"GT file not found: {gt_path}")
                stats['edge_cases']['missing_gt'] += 1
                continue
                
            out_img_path = os.path.join(output_dir, f'images/{split}', f"{city}_{base_name}.png")
            out_mask_path = os.path.join(output_dir, f'masks/{split}', f"{city}_{base_name}.png")
            
            success = process_image(
                os.path.join(city_img_dir, img_file),
                gt_path,
                out_img_path,
                out_mask_path,
                stats
            )
            
            if success:
                count += 1
    
    stats['total_images'] = count
    logger.info(f"Processed {count} images for {split} split")
    
    # Save stats
    stats_path = os.path.join(output_dir, f'stats_{split}.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Saved stats to {stats_path}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Cityscapes Dataset Preprocessing')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Root directory of Cityscapes dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save processed data')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                       help='Dataset splits to process (default: all)')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to process per split (default: all)')
    parser.add_argument('--log_level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Set the logging level')
    
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    
    # Verify paths
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    setup_directories(args.output_dir)
    
    # Process each split
    for split in args.splits:
        logger.info(f"\nProcessing {split} split...")
        stats = process_split(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            split=split,
            sample_size=args.sample_size
        )
        
if __name__ == '__main__':
    main()