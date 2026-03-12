#!/usr/bin/env python3
"""
Leaffliction - Part 2: Data Augmentation
6 transformations: Rotation, Blur, Contrast, Scaling, Illumination, Projective

Usage:
  ./Augmentation.py <image>        -> augment a single image (6 outputs)
  ./Augmentation.py <directory>    -> balance all classes in the dataset
"""

import sys
import os
import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


# ─────────────────────────────────────────────
# 6 Augmentation functions
# ─────────────────────────────────────────────

def rotation(img):
    """Rotate 45 degrees."""
    return img.rotate(45, expand=True)


def blur(img):
    """Gaussian blur."""
    return img.filter(ImageFilter.GaussianBlur(radius=3))


def contrast(img):
    """Increase contrast."""
    return ImageEnhance.Contrast(img).enhance(2.0)


def scaling(img):
    """Zoom in 130% then crop back to original size."""
    w, h = img.size
    new_w, new_h = int(w * 1.3), int(h * 1.3)
    scaled = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top  = (new_h - h) // 2
    return scaled.crop((left, top, left + w, top + h))


def illumination(img):
    """Brighten the image."""
    return ImageEnhance.Brightness(img).enhance(1.8)


def projective(img):
    """Projective (perspective) transform — trapezoid warp."""
    w, h = img.size
    offset = int(w * 0.2)
    src = [(0, 0),          (w, 0),      (w, h),      (0, h)]
    dst = [(offset, 0),     (w, 0),      (w - offset, h), (0, h)]
    coeffs = _find_coeffs(src, dst)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def _find_coeffs(src, dst):
    matrix = []
    for s, t in zip(src, dst):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(src).reshape(8)
    return np.array(np.linalg.lstsq(A, B, rcond=None)[0]).flatten()


AUGMENTATIONS = [
    ("Rotation",    rotation),
    ("Blur",        blur),
    ("Contrast",    contrast),
    ("Scaling",     scaling),
    ("Illumination",illumination),
    ("Projective",  projective),
]
AUG_NAMES = [name for name, _ in AUGMENTATIONS]
AUG_FUNCS = {name: fn for name, fn in AUGMENTATIONS}


# ─────────────────────────────────────────────
# Single image mode
# ─────────────────────────────────────────────

def augment_image(image_path, output_dir=None):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path).convert("RGB")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    save_dir = output_dir if output_dir else (os.path.dirname(image_path) or ".")
    saved = []
    for name, fn in AUGMENTATIONS:
        out_path = os.path.join(save_dir, f"{base_name}_{name}{ext}")
        fn(img).save(out_path)
        saved.append(out_path)
        print(f"  Saved: {out_path}")
    return saved


# ─────────────────────────────────────────────
# Dataset balancing mode
# ─────────────────────────────────────────────

def is_augmented(filename):
    name = os.path.splitext(filename)[0]
    return any(name.endswith(f"_{aug}") for aug in AUG_NAMES)


def balance_dataset(dataset_dir):
    if not os.path.isdir(dataset_dir):
        raise NotADirectoryError(f"Not a directory: {dataset_dir}")

    # Collect only original images (skip already-augmented ones)
    class_images = {}
    for root, dirs, files in os.walk(dataset_dir):
        imgs = [
            os.path.join(root, f) for f in sorted(files)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            and not is_augmented(f)
        ]
        if imgs:
            class_images[root] = imgs

    if not class_images:
        print("No images found.")
        return

    max_count = max(len(v) for v in class_images.values())
    print(f"\nDataset summary (target per class: {max_count})")
    print("-" * 60)
    for cls, imgs in sorted(class_images.items()):
        label = os.path.relpath(cls, dataset_dir)
        print(f"  {label:45s} {len(imgs):5d} images")

    print("\nBalancing...")

    for cls, imgs in class_images.items():
        needed = max_count - len(imgs)
        if needed <= 0:
            continue
        label = os.path.relpath(cls, dataset_dir)
        print(f"\n[{label}] needs {needed} more images")

        # Interleave: img1_Rotation, img1_Blur, ..., img1_Projective,
        #             img2_Rotation, img2_Blur, ..., img2_Projective, ...
        # All 6 augmentation types appear before any source image repeats.
        candidates = [
            (src, aug_name)
            for src in imgs
            for aug_name in AUG_NAMES
        ]

        generated = 0
        idx = 0
        while generated < needed:
            src, aug_name = candidates[idx % len(candidates)]
            idx += 1
            base_name = os.path.splitext(os.path.basename(src))[0]
            ext = os.path.splitext(src)[1]
            out_path = os.path.join(cls, f"{base_name}_{aug_name}{ext}")
            if not os.path.exists(out_path):
                AUG_FUNCS[aug_name](Image.open(src).convert("RGB")).save(out_path)
                generated += 1
                print(f"  + {os.path.basename(out_path)}")

    print(f"\nDone. All classes balanced to {max_count} images.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Leaffliction - Data Augmentation (Part 2)"
    )
    parser.add_argument(
        "path",
        help="Image file to augment, or dataset directory to balance"
    )
    parser.add_argument(
        "--output-dir", metavar="DIR",
        help="Where to save augmented images (single image mode only)"
    )
    args = parser.parse_args()

    if os.path.isdir(args.path):
        balance_dataset(args.path)
    elif os.path.isfile(args.path):
        print(f"Augmenting: {args.path}")
        augment_image(args.path, args.output_dir)
        print("Done.")
    else:
        print(f"Error: '{args.path}' is neither a file nor a directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
