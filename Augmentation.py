#!/usr/bin/env python3
import argparse
import sys
import random
import numpy as np
from pathlib import Path
from typing import Callable, Iterable, Optional, Set, Tuple, List

try:
    import cv2
except ImportError:
    print("OpenCV (cv2) is not installed. Install with: pip install opencv-python", file=sys.stderr)
    sys.exit(1)

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
KNOWN_TAGS: Set[str] = {"flip", "rotate", "contrast", "blur", "skew", "shear"}

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS

def has_any_augmentation_tag(stem: str) -> bool:
    return any(stem.endswith(f"_{tag}") for tag in KNOWN_TAGS)

def build_output_path(input_path: Path, tag: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{tag}{input_path.suffix}")

def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    if root.is_file():
        if is_image_file(root):
            yield root
        return
    if recursive:
        for p in root.rglob("*"):
            if is_image_file(p):
                yield p
    else:
        for p in root.glob("*"):
            if is_image_file(p):
                yield p

def count_images(root: Path, recursive: bool) -> int:
    return sum(1 for _ in iter_images(root, recursive))

def write_image(path: Path, img) -> None:
    params: List[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    if not cv2.imwrite(str(path), img, params):
        raise IOError(f"Failed to write output: {path}")

class ImageAugmentation:
    @staticmethod
    def load_image(image_path: Path):
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        return image

    # ------------ Transformations ------------
    # Each returns (tag, image)

    def transform_flip(img) -> Tuple[str, any]:
        """
        Random flip: h (horizontal), v (vertical), hv (both)
        """
        flip_map = {"h": 1, "v": 0, "hv": -1}
        random.seed()
        random_flip = random.choice(list(flip_map.keys()))
        flipped = cv2.flip(img, flip_map[random_flip])
        tag = "flip"
        return tag, flipped

    def transform_rotate(img) -> Tuple[str, any]:
        """
        Random rotate: 90, 180, 270 (counter-clockwise)
        """
        random.seed()
        degrees = random.choice([90, 180, 270])
        if degrees == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif degrees == 180:
            rotated = cv2.rotate(img, cv2.ROTATE_180)
        else:  # 270
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        tag = "rotate"
        return tag, rotated

    def transform_contrast(img, alpha: float = 1.5, beta: int = 0) -> Tuple[str, any]:
        """
        alpha: contrast control (1.0-3.0)
        beta: brightness control (0-100)
        """
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        tag = "contrast"
        return tag, adjusted

    def transform_skew(img, max_skew: float = 0.2) -> Tuple[str, any]:
        """
        Apply random skew to the image.
        max_skew: maximum skew factor (0.0-0.5)
        """
        h, w = img.shape[:2]
        skew_x = random.uniform(-max_skew, max_skew) * w
        skew_y = random.uniform(-max_skew, max_skew) * h

        pts1 = np.float32([[0, 0], [w, 0], [0, h]])
        pts2 = np.float32([[skew_x, skew_y],
                           [w + skew_x, skew_y],
                           [skew_x, h + skew_y]])

        M = cv2.getAffineTransform(pts1, pts2)
        skewed = cv2.warpAffine(img, M, (w, h))
        tag = "skew"
        return tag, skewed
    
    def transform_blur(img, ksize: int = 8) -> Tuple[str, any]:
        """
        Apply Gaussian blur to the image.
        ksize: kernel size (must be odd)
        """
        if ksize % 2 == 0:
            ksize += 1  # make it odd
        blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
        tag = "blur"
        return tag, blurred
    
    def transformation_shear(img, shear_factor: float = 0.2) -> Tuple[str, any]:
        """
        Apply shear transformation to the image.
        shear_factor: factor by which to shear the image
        """
        h, w = img.shape[:2]
        M = np.float32([[1, shear_factor, 0],
                        [0, 1, 0]])
        sheared = cv2.warpAffine(img, M, (w, h))
        tag = "shear"
        return tag, sheared
    
# Apply all transforms once per file
def get_all_transformers() -> List[Callable[[any], Tuple[str, any]]]:
    return [
        ImageAugmentation.transform_flip,
        ImageAugmentation.transform_rotate,
        ImageAugmentation.transform_contrast,
        ImageAugmentation.transform_skew,
        ImageAugmentation.transform_blur,
        ImageAugmentation.transformation_shear,
    ]

def apply_all_transforms_for_file(
    input_path: Path,
    overwrite: bool,
) -> int:
    """
    Apply all transforms to one input image.
    Returns number of new files created for this input.
    """
    if has_any_augmentation_tag(input_path.stem):
        return 0  # skip already transformed inputs

    img = ImageAugmentation.load_image(input_path)
    if img is None:
        print(f"Warning: cannot read image: {input_path}", file=sys.stderr)
        return 0

    created = 0
    for tf in get_all_transformers():
        tag, out_img = tf(img)
        out_path = build_output_path(input_path, tag)
        if out_path.exists() and not overwrite:
            continue
        write_image(out_path, out_img)
        print(f"Saved: {out_path}")
        created += 1
    return created

# ------------ CLI ------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply all image augmentations to a dataset until a target count is reached."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        type=str,
        help="Path to the dataset directory or image. Defaults to current directory.",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs (default: skip).",
    )
    parser.add_argument(
        "--target",
        type=int,
        help="Stop once total image count in path reaches this number (checked after finishing each file).",
    )
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    target_path = Path(args.path).resolve()
    if not target_path.exists():
        print(f"Error: '{target_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Count current total images
    current_total = count_images(target_path, recursive=args.recursive)
    if args.target is not None and current_total >= args.target:
        print(f"Target already met. Current images: {current_total} >= target {args.target}. Nothing to do.")
        return

    created_total = 0
    skipped_inputs = 0

    for img_path in iter_images(target_path, recursive=args.recursive):
        if has_any_augmentation_tag(img_path.stem):
            skipped_inputs += 1
            continue

        created_for_file = apply_all_transforms_for_file(img_path, overwrite=args.overwrite)
        created_total += created_for_file
        current_total += created_for_file  # increase only after finishing all transforms for this file

        if args.target is not None and current_total >= args.target:
            break

    print(f"Done. Created: {created_total}, Skipped inputs: {skipped_inputs}, Total images now: {current_total}")

if __name__ == "__main__":
    main()