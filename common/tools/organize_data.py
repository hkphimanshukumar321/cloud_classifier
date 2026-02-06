# -*- coding: utf-8 -*-
# ==============================================================================
# Copyright (c) 2026 Himanshu Kumar, IIT Bhubaneswar. All Rights Reserved.
# Email: hkphimanshukumar321@gmail.com
# ==============================================================================

"""
Universal Data Organizer
========================

A unified tool to ingest and organize datasets for:
1. Classification (Folders)
2. Segmentation (Images + Masks)
3. Detection (Images + YOLO/XML labels)

Usage:
    python common/tools/organize_data.py --task classification --images raw_imgs/ --csv labels.csv
    python common/tools/organize_data.py --task segmentation --images raw_imgs/ --masks raw_masks/
"""

import sys
import shutil
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional

def organize_classification(
    images_dir: Path,
    output_dir: Path,
    metadata_path: Optional[Path] = None,
    file_col: str = 'filename',
    label_col: str = 'label',
    delimiter: Optional[str] = None,
    index: int = 0
):
    """Organize images into class folders."""
    output_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = None
    
    # 1. Load from Metadata
    if metadata_path and metadata_path.exists():
        print(f"[*] Loading metadata from {metadata_path}...")
        if metadata_path.suffix == '.csv':
            df = pd.read_csv(metadata_path)
        elif metadata_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(metadata_path)
        else:
            print("[!] Unsupported metadata format")
            return

    # 2. Load from Filenames
    elif delimiter:
        print(f"[*] Inferring labels (delimiter='{delimiter}', index={index})...")
        data = []
        for p in images_dir.glob("*"):
            if p.is_file() and p.suffix.lower() in {'.jpg', '.png', '.jpeg'}:
                try:
                    label = p.stem.split(delimiter)[index]
                    data.append({file_col: p.name, label_col: label})
                except IndexError:
                    pass
        df = pd.DataFrame(data)
    
    if df is None:
        print("[!] Error: No valid metadata or filename pattern provided.")
        return

    # Execute Copy
    print(f"[*] Moving {len(df)} images to {output_dir}...")
    count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fname = str(row[file_col])
        label = str(row[label_col]).strip()
        
        src = images_dir / fname
        dst = output_dir / label / fname
        
        if src.exists():
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dst)
            count += 1
            
    print(f"✅ Classsification: Organized {count} images.")


def organize_paired_data(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    task_type: str  # 'segmentation' or 'detection'
):
    """Organize paired data (Image + Mask/Label)."""
    
    # Define subfolders based on task
    if task_type == 'segmentation':
        # Segmentation: images/ + masks/
        out_imgs = output_dir / "images"
        out_lbls = output_dir / "masks"
        valid_lbl_exts = {'.png', '.jpg', '.bmp', '.tif', '.tiff'} # Masks are images
    else:
        # Detection: images/ + labels/
        out_imgs = output_dir / "images"
        out_lbls = output_dir / "labels"
        valid_lbl_exts = {'.txt', '.xml', '.json'} # YOLO/Pascal/COCO
        
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Pairing images from {images_dir} with labels from {labels_dir}...")
    
    # Scan for pairs
    paired = 0
    missing = 0
    
    # Common image extensions
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    for img_path in images_dir.glob("*"):
        if img_path.suffix.lower() not in img_exts:
            continue
            
        # Try to find matching label with ANY valid extension
        found_label = False
        for lbl_ext in valid_lbl_exts:
            lbl_path = labels_dir / (img_path.stem + lbl_ext)
            if lbl_path.exists():
                # COPY PAIR
                shutil.copy2(img_path, out_imgs / img_path.name)
                shutil.copy2(lbl_path, out_lbls / lbl_path.name)
                paired += 1
                found_label = True
                break
        
        if not found_label:
            # Check edge case: filename.jpg -> filename.txt (sometimes simpler match)
             lbl_path = labels_dir / (img_path.name + '.txt') # simplistic check
             pass # Logic mostly covered above
             missing += 1
             
    print(f"✅ {task_type.capitalize()}: Paired {paired} samples.")
    print(f"❌ Unmatched images: {missing}")
    print(f"📂 Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omni Workbench Data Ingestion")
    parser.add_argument("--task", required=True, choices=['classification', 'segmentation', 'detection'])
    parser.add_argument("--images", required=True, help="Source images folder")
    parser.add_argument("--output", default="../../data", help="Output root (default: omni_workbench_MLsuite/data)")
    
    # Classification specific
    parser.add_argument("--csv", help="Labels CSV/Excel")
    parser.add_argument("--delimiter", help="Filename delimiter for label inference")
    parser.add_argument("--index", type=int, default=0, help="Index of label in filename split")
    parser.add_argument("--file_col", default="filename")
    parser.add_argument("--label_col", default="label")
    
    # Segmentation/Detection specific
    parser.add_argument("--labels", help="Source folder for masks (Seg) or text labels (Det)")
    
    args = parser.parse_args()
    
    # Determine absolute output path task subfolder
    root_out = Path(args.output)
    task_out = root_out / args.task
    
    if args.task == 'classification':
        organize_classification(
            Path(args.images), 
            task_out, 
            Path(args.csv) if args.csv else None,
            args.file_col, args.label_col,
            args.delimiter, args.index
        )
    else:
        # Segmentation or Detection
        if not args.labels:
            print(f"[!] Error: --labels folder required for {args.task}")
            sys.exit(1)
            
        organize_paired_data(
            Path(args.images),
            Path(args.labels),
            task_out,
            args.task
        )
