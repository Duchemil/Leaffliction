#!/usr/bin/env python3
"""
Leaf Disease Classifier - Prediction
--------------------------------------
Usage:
    # Image unique
    python predict.py "./Apple/apple_healthy/image (1).JPG"

    # Plusieurs images
    python predict.py img1.jpg img2.jpg img3.jpg

    # Répertoire (récursif par défaut)
    python predict.py ./Apple/apple_healthy/

    # Répertoire avec sous-répertoires
    python predict.py ./Apple/ --recursive

    # Sauvegarder les résultats au lieu d'afficher
    python predict.py ./Apple/ --recursive --save ./results/

Arguments:
    sources          Un ou plusieurs fichiers / répertoires
    -model PATH      Chemin vers le zip du modèle (défaut: model.zip)
    --recursive      Parcourir les sous-répertoires (défaut si src est un dossier)
    --save DIR       Sauvegarder les résultats dans DIR au lieu d'afficher
    --no-display     Ne pas afficher les fenêtres (utile avec --save)
    -h               Aide
"""

import sys
import os
import argparse
import zipfile
import json
import tempfile
import shutil
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ── Auto-select a working matplotlib backend ──
import matplotlib
def _set_backend():
    for backend in ('TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'wxAgg'):
        try:
            matplotlib.use(backend)
            import matplotlib.figure as _fig
            _fig.Figure()
            return backend
        except Exception:
            continue
    matplotlib.use('Agg')
    return 'Agg'

_BACKEND = _set_backend()
import matplotlib.pyplot as plt

# Prevent OpenCV's Qt plugin from conflicting
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import cv2
if os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
    del os.environ['QT_QPA_PLATFORM']

import tensorflow as tf

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif',
                    '.JPG', '.JPEG', '.PNG'}


# ─────────────────────────────────────────────
#  File discovery
# ─────────────────────────────────────────────

def collect_images(sources, recursive=True):
    """
    From a list of files/directories, return a flat list of image paths.
    Each entry is (image_path, label_hint) where label_hint is the
    parent folder name (= ground truth class if folder is named after it).
    """
    collected = []

    for src in sources:
        if os.path.isfile(src):
            ext = os.path.splitext(src)[1]
            if ext in IMAGE_EXTENSIONS:
                label = os.path.basename(os.path.dirname(src))
                collected.append((src, label))
            else:
                print(f"  Warning: skipping non-image file: {src}")

        elif os.path.isdir(src):
            if recursive:
                for root, dirs, files in os.walk(src):
                    dirs.sort()
                    for fname in sorted(files):
                        if os.path.splitext(fname)[1] in IMAGE_EXTENSIONS:
                            full = os.path.join(root, fname)
                            label = os.path.basename(root)
                            collected.append((full, label))
            else:
                for fname in sorted(os.listdir(src)):
                    if os.path.splitext(fname)[1] in IMAGE_EXTENSIONS:
                        full = os.path.join(src, fname)
                        label = os.path.basename(src)
                        collected.append((full, label))
        else:
            print(f"  Warning: path not found: {src}")

    return collected


# ─────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────

def load_model_from_zip(zip_path):
    """Extract and load model from zip archive."""
    if not os.path.exists(zip_path):
        print(f"Error: model zip not found: '{zip_path}'")
        print("Run train.py first to generate the model.")
        sys.exit(1)

    tmp_dir = tempfile.mkdtemp(prefix='leaf_predict_')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_dir)

    model   = tf.keras.models.load_model(os.path.join(tmp_dir, 'model.h5'))
    classes_path  = os.path.join(tmp_dir, 'classes.txt')
    metrics_path  = os.path.join(tmp_dir, 'metrics.json')

    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]

    img_size = 224
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            img_size = json.load(f).get('img_size', 224)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return model, classes, img_size


# ─────────────────────────────────────────────
#  Image transformation
# ─────────────────────────────────────────────

def transform_image(img):
    """Apply leaf mask pipeline, return cleaned image (white background)."""
    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_s  = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(gray_s, (11, 11), 0)
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    kernel  = np.ones((5, 5), np.uint8)
    mask    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask    = cv2.morphologyEx(mask,   cv2.MORPH_OPEN,  kernel, iterations=2)
    result  = cv2.bitwise_and(img, img, mask=mask)
    white_bg = np.full_like(img, 255)
    white_bg[mask > 0] = result[mask > 0]
    return white_bg


# ─────────────────────────────────────────────
#  Prediction
# ─────────────────────────────────────────────

def predict_image(image_path, model, classes, img_size):
    """Return (predicted_class, confidence, all_probs_dict)."""
    img  = tf.keras.preprocessing.image.load_img(
               image_path, target_size=(img_size, img_size))
    arr  = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr  = np.expand_dims(arr, 0)
    preds = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(preds))
    return classes[idx], float(preds[idx]), \
           {cls: float(preds[i]) for i, cls in enumerate(classes)}


def predict_batch(image_paths, model, classes, img_size):
    """Predict all images in one batched call for efficiency."""
    arrays = []
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(
                  path, target_size=(img_size, img_size))
        arrays.append(tf.keras.preprocessing.image.img_to_array(img) / 255.0)

    batch = np.stack(arrays, axis=0)
    all_preds = model.predict(batch, verbose=0)

    results = []
    for preds in all_preds:
        idx = int(np.argmax(preds))
        results.append((
            classes[idx],
            float(preds[idx]),
            {cls: float(preds[i]) for i, cls in enumerate(classes)}
        ))
    return results


# ─────────────────────────────────────────────
#  Display / Save
# ─────────────────────────────────────────────

def build_figure(image_path, predicted_class, confidence, all_probs, true_label=None):
    """Build and return a matplotlib figure for one prediction."""
    img_bgr   = cv2.imread(image_path)
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    trans_rgb = cv2.cvtColor(transform_image(img_bgr), cv2.COLOR_BGR2RGB)

    correct = (true_label is not None and
               true_label.lower() == predicted_class.lower())
    banner_color  = '#003300' if correct else '#1a1a1a'
    class_color   = '#00FF88' if correct else '#FF6666'

    fig = plt.figure(figsize=(14, 7), facecolor='black')

    ax1 = fig.add_axes([0.02, 0.20, 0.44, 0.75])
    ax2 = fig.add_axes([0.54, 0.20, 0.44, 0.75])
    ax1.imshow(img_rgb);   ax1.axis('off')
    ax2.imshow(trans_rgb); ax2.axis('off')

    ax1.set_title(os.path.basename(image_path),
                  color='white', fontsize=8, pad=4)

    ax_b = fig.add_axes([0.0, 0.0, 1.0, 0.18], facecolor=banner_color)
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1); ax_b.axis('off')
    ax_b.axhline(y=0.95, color='#444444', linewidth=1)

    ax_b.text(0.08, 0.65, '===', color='white',
              fontsize=18, fontweight='bold', ha='center', va='center')
    ax_b.text(0.50, 0.65, 'DL classification', color='white',
              fontsize=20, fontweight='bold', ha='center', va='center')
    ax_b.text(0.92, 0.65, '===', color='white',
              fontsize=18, fontweight='bold', ha='center', va='center')

    ax_b.text(0.28, 0.20, 'Class predicted :', color='white',
              fontsize=14, fontweight='bold', ha='center', va='center')
    ax_b.text(0.65, 0.20, predicted_class, color=class_color,
              fontsize=14, fontweight='bold', ha='center', va='center')

    if true_label:
        status = '✓ Correct' if correct else f'✗ Expected: {true_label}'
        ax_b.text(0.50, 0.50, status,
                  color='#00FF88' if correct else '#FF4444',
                  fontsize=9, ha='center', va='center')

    plt.suptitle(f'Confidence: {confidence*100:.1f}%',
                 color='#AAAAAA', fontsize=10, y=0.205)
    return fig


def print_result(image_path, predicted_class, confidence, all_probs,
                 true_label=None, index=None, total=None):
    prefix = f"[{index}/{total}] " if index and total else ""
    correct = (true_label and true_label.lower() == predicted_class.lower())
    marker  = '✓' if correct else ('✗' if true_label else ' ')
    print(f"\n  {prefix}{marker} {os.path.basename(image_path)}")
    print(f"     Predicted : {predicted_class}  ({confidence*100:.1f}%)")
    if true_label and not correct:
        print(f"     Expected  : {true_label}")
    top = sorted(all_probs.items(), key=lambda x: -x[1])[:3]
    for cls, prob in top:
        bar = '█' * int(prob * 25)
        print(f"     {cls:<35} {prob*100:5.1f}% {bar}")


def print_summary(results_list):
    """Print accuracy summary when multiple images are processed."""
    total   = len(results_list)
    correct = sum(1 for _, pred, _, true in results_list
                  if true and true.lower() == pred.lower())

    print(f"\n{'='*55}")
    print(f"  Summary: {total} image(s) processed")
    if any(true for _, _, _, true in results_list):
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    # Per-class breakdown
    from collections import defaultdict
    per_class = defaultdict(lambda: {'total': 0, 'correct': 0})
    for _, pred, _, true in results_list:
        if true:
            per_class[true]['total'] += 1
            if true.lower() == pred.lower():
                per_class[true]['correct'] += 1

    if per_class:
        print(f"\n  Per-class accuracy:")
        for cls, counts in sorted(per_class.items()):
            acc = counts['correct'] / counts['total'] * 100
            bar = '█' * int(acc / 5)
            print(f"    {cls:<35} {counts['correct']:3}/{counts['total']:3}"
                  f"  {acc:5.1f}% {bar}")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        prog='predict.py',
        description='Predict leaf disease - single image, multiple images, or directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image unique
  python predict.py "./Apple/apple_healthy/image (1).JPG"

  # Plusieurs images
  python predict.py img1.jpg img2.jpg img3.jpg

  # Répertoire (avec sous-répertoires)
  python predict.py ./Apple/ --recursive

  # Sauvegarder les résultats sans afficher
  python predict.py ./Apple/ --recursive --save ./results/ --no-display

  # Modèle personnalisé
  python predict.py ./Apple/ -model apple_model.zip
        """
    )
    parser.add_argument('sources', nargs='+',
                        help='Image(s) ou répertoire(s) à classifier')
    parser.add_argument('-model', '--model', default='model.zip',
                        help='Chemin vers le zip du modèle (défaut: model.zip)')
    parser.add_argument('--recursive', '-r', action='store_true', default=True,
                        help='Parcourir les sous-répertoires (défaut: True)')
    parser.add_argument('--no-recursive', action='store_true',
                        help='Ne pas parcourir les sous-répertoires')
    parser.add_argument('--save', metavar='DIR', default=None,
                        help='Sauvegarder les figures dans ce répertoire')
    parser.add_argument('--no-display', action='store_true',
                        help='Ne pas afficher les fenêtres interactives')
    return parser.parse_args()


def main():
    args = parse_args()
    recursive = args.recursive and not args.no_recursive

    # Load model
    print(f"Loading model from '{args.model}'...")
    model, classes, img_size = load_model_from_zip(args.model)
    print(f"  Classes ({len(classes)}): {', '.join(classes)}")
    print(f"  Image size: {img_size}px\n")

    # Collect images
    images = collect_images(args.sources, recursive=recursive)
    if not images:
        print("Error: no images found.")
        sys.exit(1)

    print(f"Found {len(images)} image(s) to process.")

    # Batch predict
    paths  = [p for p, _ in images]
    labels = [l for _, l in images]
    preds  = predict_batch(paths, model, classes, img_size)

    # Prepare save dir
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    # Display / save results
    results_list = []
    for i, ((path, true_label), (pred_class, conf, all_probs)) in \
            enumerate(zip(images, preds), 1):

        print_result(path, pred_class, conf, all_probs,
                     true_label=true_label, index=i, total=len(images))

        results_list.append((path, pred_class, conf, true_label))

        show = not args.no_display
        save = args.save is not None

        if show or save:
            fig = build_figure(path, pred_class, conf, all_probs,
                               true_label=true_label)
            if save:
                basename = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(args.save, f"{basename}_prediction.png")
                fig.savefig(out_path, dpi=120, bbox_inches='tight',
                            facecolor='black')
                print(f"     Saved → {out_path}")
            if show:
                plt.show()
            plt.close(fig)

    # Summary (only if multiple images)
    if len(images) > 1:
        print_summary(results_list)


if __name__ == '__main__':
    main()