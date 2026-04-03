import sys
import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt


def rotation(img):
    return img.rotate(45, expand=True)


def blur(img):
    return img.filter(ImageFilter.GaussianBlur(radius=3))


def contrast(img):
    return ImageEnhance.Contrast(img).enhance(2.0)


def scaling(img):
    w, h = img.size
    new_w, new_h = int(w * 1.3), int(h * 1.3)
    scaled = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    return scaled.crop((left, top, left + w, top + h))


def illumination(img):
    return ImageEnhance.Brightness(img).enhance(1.8)


def projective(img):
    w, h = img.size
    offset = int(w * 0.2)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [(offset, 0), (w, 0), (w - offset, h), (0, h)]
    coeffs = find_coeffs(src, dst)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def find_coeffs(src, dst):
    matrix = []
    for s, t in zip(src, dst):
        matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
        matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
    A = np.matrix(matrix, dtype=np.float64)
    B = np.array(src).reshape(8)
    return np.array(np.linalg.lstsq(A, B, rcond=None)[0]).flatten()


AUGMENTATIONS = [
    ("Rotation", rotation),
    ("Blur", blur),
    ("Contrast", contrast),
    ("Scaling", scaling),
    ("Illumination", illumination),
    ("Projective", projective),
]
AUG_NAMES = [name for name, _ in AUGMENTATIONS]
AUG_FUNCS = {name: fn for name, fn in AUGMENTATIONS}


def display_augmentations(original, augmented_images, base_name):
    num_aug = len(augmented_images)
    fig, axes = plt.subplots(2, ((num_aug) // 2)+1, figsize=(12, 6))
    axes = axes.flatten()

    axes[0].imshow(original)
    axes[0].set_title("Original",  fontsize=12, fontweight='bold', pad=10)
    axes[0].axis("off")

    for idx, (name, aug_img) in enumerate(augmented_images, 1):
        axes[idx].imshow(aug_img)
        axes[idx].set_title(name, fontsize=12, fontweight='bold', pad=10)
        axes[idx].axis("off")

    for idx in range(num_aug + 1, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Augmentations: {base_name}", fontsize=16)
    plt.tight_layout()
    plt.show()


def augment_image(img_path, output_dir=None, display=False):
    if not os.path.isfile(img_path):
        print(f"Error: Image not found: {img_path}")
        exit()
    img = Image.open(img_path).convert("RGB")
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    ext = os.path.splitext(img_path)[1]
    save_dir = output_dir if output_dir else (os.path.dirname(img_path) or ".")
    saved = []

    augmented_images = []

    for name, fn in AUGMENTATIONS:
        out_path = os.path.join(save_dir, f"{base_name}_{name}{ext}")
        fn(img).save(out_path)
        saved.append(out_path)
        print(f"  Saved: {out_path}")

        if display:
            augmented_images.append((name, Image.open(out_path)))
    
    if display:
        display_augmentations(img, augmented_images, base_name)

    return saved


def is_augmented(filename):
    name = os.path.splitext(filename)[0]
    for aug in AUG_NAMES:
        if name.endswith(f"_{aug}"):
            return True
    return False


def collect_class_images(dataset_dir):
    class_images = {}
    for root, dirs, files in os.walk(dataset_dir):
        imgs = []
        for f in sorted(files):
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                if not is_augmented(f):
                    imgs.append(os.path.join(root, f))
        if imgs:
            class_images[root] = imgs
    return class_images


def print_summary(class_images, dataset_dir, max_count):
    print(f"\nDataset summary (target per class: {max_count})")
    print("-" * 60)
    for cls, imgs in sorted(class_images.items()):
        label = os.path.relpath(cls, dataset_dir)
        print(f"  {label:45s} {len(imgs):5d} images")


def generate_images(cls, imgs, needed, dataset_dir):
    label = os.path.relpath(cls, dataset_dir)
    print(f"\n[{label}] needs {needed} more images")

    candidates = []
    for src in imgs:
        for aug_name in AUG_NAMES:
            candidates.append((src, aug_name))

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


def balance_dataset(dataset_dir):
    if not os.path.isdir(dataset_dir):
        print(f"Error: Not a directory: {dataset_dir}")
        exit()

    class_images = collect_class_images(dataset_dir)

    if not class_images:
        print("No images found.")
        return

    max_count = max(len(v) for v in class_images.values())
    print_summary(class_images, dataset_dir, max_count)

    print("\nBalancing...")
    for cls, imgs in class_images.items():
        needed = max_count - len(imgs)
        if needed <= 0:
            continue
        generate_images(cls, imgs, needed, dataset_dir)

    print(f"\nDone. All classes balanced to {max_count} images.")


def main(args):
    if len(args) < 2 or len(args) > 3:
        print("Usage: ./Augmentation.py <image|directory> [output DIR]")
        exit()

    path = args[1]
    output_dir = None
    if len(args) == 3:
        output_dir = args[2]
        if not os.path.exists(output_dir):
            print(f"Error: Output doesn't exist: {output_dir}")
            exit()
        elif not os.path.isdir(output_dir):
            print(f"Error: Output is not a directory: {output_dir}")
            exit()

    if os.path.isdir(path):
        balance_dataset(path)
    elif os.path.isfile(path):
        print(f"Augmenting: {path}")
        augment_image(path, output_dir, display=True)
        print("Done.")
    else:
        print(f"Error: '{path}' is neither a file nor a directory.")
        exit()


if __name__ == "__main__":
    main(sys.argv)
