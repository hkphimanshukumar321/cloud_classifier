#!/usr/bin/env python3
"""
Offline Data Augmentation & Class Balancing
============================================

Reads images from data/classification/raw,
generates augmented copies to balance all classes to the majority class count,
and saves everything (originals + augmented) to data/classification/raw2.

Usage:
    python common/tools/augment_and_balance.py
    python common/tools/augment_and_balance.py --target 500
    python common/tools/augment_and_balance.py --src data/classification/raw --dst data/classification/raw2
"""

import os
import sys
import random
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ─── Augmentation Functions ───────────────────────────────────────────────────

def random_flip(img: np.ndarray) -> np.ndarray:
    """Random horizontal and/or vertical flip."""
    if random.random() > 0.5:
        img = cv2.flip(img, 1)  # horizontal
    if random.random() > 0.5:
        img = cv2.flip(img, 0)  # vertical (clouds have no strict up)
    return img


def random_rotation(img: np.ndarray) -> np.ndarray:
    """Random rotation by 0, 90, 180, or 270 degrees."""
    k = random.randint(0, 3)
    return np.rot90(img, k)


def random_brightness(img: np.ndarray, delta: float = 40.0) -> np.ndarray:
    """Random brightness shift."""
    shift = random.uniform(-delta, delta)
    return np.clip(img.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def random_contrast(img: np.ndarray, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    """Random contrast adjustment."""
    factor = random.uniform(low, high)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    return np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)


def random_saturation(img: np.ndarray, low: float = 0.7, high: float = 1.3) -> np.ndarray:
    """Random saturation adjustment (HSV space)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(low, high)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def random_blur(img: np.ndarray) -> np.ndarray:
    """Light Gaussian blur (simulates atmospheric haze)."""
    if random.random() > 0.5:
        ksize = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img


def random_noise(img: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Add slight Gaussian noise."""
    if random.random() > 0.5:
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply a random combination of augmentations."""
    img = random_flip(img)
    img = random_rotation(img)
    img = random_brightness(img)
    img = random_contrast(img)
    img = random_saturation(img)
    img = random_blur(img)
    img = random_noise(img)
    return img


# ─── Main Logic ──────────────────────────────────────────────────────────────

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}


def get_class_info(src_dir: Path) -> dict:
    """Get class names and file lists."""
    classes = {}
    for class_dir in sorted(src_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        files = sorted([
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])
        classes[class_dir.name] = files
    return classes


def augment_and_balance(
    src_dir: Path,
    dst_dir: Path,
    target_per_class: int = None,
    seed: int = 42
):
    """
    Balance dataset by augmenting minority classes.
    
    1. Copies ALL original images to dst_dir
    2. For classes below target_per_class, generates augmented copies
    3. Target defaults to the majority class count
    """
    random.seed(seed)
    np.random.seed(seed)

    classes = get_class_info(src_dir)
    if not classes:
        print(f"❌ No class folders found in {src_dir}")
        return

    counts = {name: len(files) for name, files in classes.items()}
    max_count = max(counts.values())
    target = target_per_class or max_count

    print("=" * 60)
    print("OFFLINE AUGMENTATION & BALANCING")
    print("=" * 60)
    print(f"\n📂 Source:  {src_dir}")
    print(f"📂 Output:  {dst_dir}")
    print(f"🎯 Target per class: {target}")
    print(f"\n{'Class':<15} {'Original':>10} {'To Generate':>12} {'Final':>8}")
    print("-" * 50)
    for name, cnt in counts.items():
        to_gen = max(0, target - cnt)
        print(f"  {name:<13} {cnt:>10} {to_gen:>12} {cnt + to_gen:>8}")
    total_orig = sum(counts.values())
    total_aug = sum(max(0, target - c) for c in counts.values())
    print("-" * 50)
    print(f"  {'TOTAL':<13} {total_orig:>10} {total_aug:>12} {total_orig + total_aug:>8}")

    # Clean output directory
    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    # Process each class
    print(f"\n🔄 Processing...")
    for class_name, files in classes.items():
        out_class_dir = dst_dir / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        orig_count = len(files)
        needed = max(0, target - orig_count)

        # Step 1: Copy all originals
        for f in tqdm(files, desc=f"  {class_name} (copy)", leave=False):
            shutil.copy2(f, out_class_dir / f.name)

        # Step 2: Generate augmented images to fill the gap
        if needed > 0:
            for i in tqdm(range(needed), desc=f"  {class_name} (augment)", leave=False):
                # Pick a random original to augment
                src_file = random.choice(files)
                img = cv2.imread(str(src_file))
                if img is None:
                    continue

                aug_img = augment_image(img)

                # Save with unique name
                stem = src_file.stem
                ext = src_file.suffix
                aug_name = f"{stem}_aug{i:04d}{ext}"
                cv2.imwrite(str(out_class_dir / aug_name), aug_img)

        final_count = len(list(out_class_dir.iterdir()))
        print(f"  ✅ {class_name}: {orig_count} original + {needed} augmented = {final_count} total")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"🎉 DONE! Balanced dataset saved to: {dst_dir}")
    
    # Verify
    print(f"\n📊 Verification:")
    final_classes = get_class_info(dst_dir)
    for name, files in final_classes.items():
        print(f"  {name:<15}: {len(files):>5} images")
    total_final = sum(len(f) for f in final_classes.values())
    print(f"  {'TOTAL':<15}: {total_final:>5} images")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline augmentation & class balancing for cloud density dataset"
    )
    parser.add_argument(
        "--src", type=str,
        default="data/classification/raw",
        help="Source directory with class subfolders"
    )
    parser.add_argument(
        "--dst", type=str,
        default="data/classification/raw2",
        help="Output directory for balanced dataset"
    )
    parser.add_argument(
        "--target", type=int, default=None,
        help="Target images per class (default: match majority class)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    augment_and_balance(
        src_dir=Path(args.src),
        dst_dir=Path(args.dst),
        target_per_class=args.target,
        seed=args.seed
    )
