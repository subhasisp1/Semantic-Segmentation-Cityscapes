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
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(_name_)

# Class mappings (same as before)
CITYSCAPES_ID_TO_NAME = {...}  # Omitted for brevity
NEW_MAPPING = {...}  # Omitted for brevity
NEW_CLASS_NAMES = {...}  # Omitted for brevity

# Target size for resizing
TARGET_SIZE = (256, 256)

def setup_directories(output_dir: str) -> None:
    os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images/test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks/test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images_processed/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks_processed/train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images_processed/val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks_processed/val'), exist_ok=True)
    logger.info(f"Created directory structure in {output_dir}")

def process_image(
    img_path: str,
    gt_path: str,
    output_img_path: str,
    output_mask_path: str,
    processed_img_path: str,
    processed_mask_path: str,
    stats: Dict
) -> bool:
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

        # Resize and normalize
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_np, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        img_normalized = img_resized / 255.0
        np.save(processed_img_path.replace(".png", ".npy"), img_normalized)

        mask_resized = cv2.resize(new_mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        Image.fromarray(mask_resized).save(processed_mask_path)

        return True

    except Exception as e:
        logger.error(f"Error processing {img_path}: {str(e)}")
        stats['edge_cases']['processing_error'] += 1
        return False

def process_split(data_dir: str, output_dir: str, split: str = 'train', sample_size: Optional[int] = None) -> Dict:
    stats = {'edge_cases': defaultdict(int), 'class_pixels': defaultdict(int), 'total_images': 0}
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
            out_proc_img_path = os.path.join(output_dir, f'images_processed/{split}', f"{city}_{base_name}.png")
            out_proc_mask_path = os.path.join(output_dir, f'masks_processed/{split}', f"{city}_{base_name}.png")

            success = process_image(
                os.path.join(city_img_dir, img_file),
                gt_path,
                out_img_path,
                out_mask_path,
                out_proc_img_path,
                out_proc_mask_path,
                stats
            )

            if success:
                count += 1

    stats['total_images'] = count
    logger.info(f"Processed {count} images for {split} split")

    stats_path = os.path.join(output_dir, f'stats_{split}.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Saved stats to {stats_path}")

    return stats

def main():
    parser = argparse.ArgumentParser(description='Cityscapes Dataset Preprocessing')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory of Cityscapes dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'], help='Dataset splits to process')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to process per split')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Set the logging level')
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    setup_directories(args.output_dir)

    for split in args.splits:
        logger.info(f"\nProcessing {split} split...")
        process_split(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            split=split,
            sample_size=args.sample_size
        )

if _name_ == '_main_':
    main()
