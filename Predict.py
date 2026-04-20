import sys
import os
import zipfile
import json
import tempfile
import shutil
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

matplotlib.use('TkAgg')

IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'
}


def collect_images(sources, recursive=True):
    collected = []
    for src in sources:
        if os.path.isfile(src):
            if os.path.splitext(src)[1] in IMAGE_EXTENSIONS:
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
                            collected.append((
                                os.path.join(root, fname),
                                os.path.basename(root)
                            ))
            else:
                for fname in sorted(os.listdir(src)):
                    if os.path.splitext(fname)[1] in IMAGE_EXTENSIONS:
                        collected.append((
                            os.path.join(src, fname),
                            os.path.basename(src)
                        ))
        else:
            print(f"  Warning: path not found: {src}")
    return collected


def load_model_from_zip(zip_path):
    if not os.path.exists(zip_path):
        print(f"Error: model zip not found: '{zip_path}'")
        print("Run Train.py first to generate the model.")
        exit()

    tmp_dir = tempfile.mkdtemp(prefix='leaf_predict_')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(tmp_dir)

    model = tf.keras.models.load_model(os.path.join(tmp_dir, 'model.h5'))
    with open(os.path.join(tmp_dir, 'classes.txt'), 'r') as f:
        classes = [line.strip() for line in f if line.strip()]

    img_size = 224
    metrics_path = os.path.join(tmp_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            img_size = json.load(f).get('img_size', 224)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return model, classes, img_size


def transform_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_s = hsv[:, :, 1]
    blurred = cv2.GaussianBlur(gray_s, (11, 11), 0)
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    result = cv2.bitwise_and(img, img, mask=mask)
    white_bg = np.full_like(img, 255)
    white_bg[mask > 0] = result[mask > 0]
    return white_bg


def predict_batch(image_paths, model, classes, img_size):
    arrays = []
    for path in image_paths:
        img = tf.keras.preprocessing.image.load_img(
            path, target_size=(img_size, img_size))
        arrays.append(
            tf.keras.preprocessing.image.img_to_array(img) / 255.0
        )
    batch = np.stack(arrays, axis=0)
    all_preds = model.predict(batch, verbose=0)
    results = []
    for preds in all_preds:
        idx = int(np.argmax(preds))
        probs = {}
        for i, cls in enumerate(classes):
            probs[cls] = float(preds[i])
        results.append((classes[idx], float(preds[idx]), probs))
    return results


def build_figure(image_path, predicted_class, confidence, all_probs,
                 true_label=None):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    trans_rgb = cv2.cvtColor(transform_image(img_bgr), cv2.COLOR_BGR2RGB)

    correct = (true_label is not None and
               true_label.lower() == predicted_class.lower())
    banner_color = '#003300' if correct else '#1a1a1a'
    class_color = '#00FF88' if correct else '#FF6666'

    fig = plt.figure(figsize=(14, 7), facecolor='black')
    ax1 = fig.add_axes([0.02, 0.20, 0.44, 0.75])
    ax2 = fig.add_axes([0.54, 0.20, 0.44, 0.75])
    ax1.imshow(img_rgb)
    ax1.axis('off')
    ax2.imshow(trans_rgb)
    ax2.axis('off')
    ax1.set_title(os.path.basename(image_path), color='white',
                  fontsize=8, pad=4)

    ax_b = fig.add_axes([0.0, 0.0, 1.0, 0.18], facecolor=banner_color)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis('off')
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
        status = 'Correct' if correct else f'Expected: {true_label}'
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
    marker = 'OK' if correct else ('ERR' if true_label else '   ')
    print(f"\n  {prefix}[{marker}] {os.path.basename(image_path)}")
    print(f"     Predicted : {predicted_class}  ({confidence*100:.1f}%)")
    if true_label and not correct:
        print(f"     Expected  : {true_label}")
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1])[:3]:
        print(f"     {cls:<35} {prob*100:5.1f}%")


def print_summary(results_list):
    from collections import defaultdict

    total = len(results_list)
    correct = 0
    for _, pred, _, true in results_list:
        if true and true.lower() == pred.lower():
            correct += 1

    print(f"\n{'='*55}")
    print(f"  Summary: {total} image(s) processed")

    has_labels = any(true for _, _, _, true in results_list)
    if has_labels:
        print(f"  Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    per_class = defaultdict(lambda: {'total': 0, 'correct': 0})
    for _, pred, _, true in results_list:
        if true:
            per_class[true]['total'] += 1
            if true.lower() == pred.lower():
                per_class[true]['correct'] += 1

    if per_class:
        print("\n  Per-class accuracy:")
        for cls, counts in sorted(per_class.items()):
            acc = counts['correct'] / counts['total'] * 100
            bar = '#' * int(acc / 5)
            print(f"    {cls:<35} {counts['correct']:3}/{counts['total']:3}"
                  f"  {acc:5.1f}% {bar}")
    print(f"{'='*55}\n")


def parse_args(args):
    sources = []
    model_path = 'model.zip'
    recursive = True
    save_dir = None
    no_display = False

    i = 1
    while i < len(args):
        if args[i] in ('-model', '--model'):
            model_path = args[i + 1]
            i += 2
        elif args[i] in ('--recursive', '-r'):
            recursive = True
            i += 1
        elif args[i] == '--no-recursive':
            recursive = False
            i += 1
        elif args[i] == '--save':
            save_dir = args[i + 1]
            i += 2
        elif args[i] == '--no-display':
            no_display = True
            i += 1
        elif args[i] in ('-h', '--help'):
            print(__doc__)
            exit()
        else:
            sources.append(args[i])
            i += 1

    return sources, model_path, recursive, save_dir, no_display


def main(args):
    if len(args) < 2:
        print("Usage: python Predict.py <image|directory> [options]")
        print("Use -h for help.")
        exit()

    sources, model_path, recursive, save_dir, no_display = parse_args(args)

    print(f"Loading model from '{model_path}'...")
    model, classes, img_size = load_model_from_zip(model_path)
    print(f"  Classes ({len(classes)}): {', '.join(classes)}")
    print(f"  Image size: {img_size}px\n")

    images = collect_images(sources, recursive=recursive)
    if not images:
        print("Error: no images found.")
        exit()

    print(f"Found {len(images)} image(s) to process.")

    paths = [p for p, _ in images]
    preds = predict_batch(paths, model, classes, img_size)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results_list = []
    for i, ((path, true_label), (pred_class, conf, all_probs)) in \
            enumerate(zip(images, preds), 1):

        print_result(path, pred_class, conf, all_probs,
                     true_label=true_label, index=i, total=len(images))
        results_list.append((path, pred_class, conf, true_label))

        show = not no_display
        save = save_dir is not None

        if show or save:
            fig = build_figure(path, pred_class, conf, all_probs,
                               true_label=true_label)
            if save:
                basename = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(save_dir, f"{basename}_prediction.png")
                fig.savefig(out_path, dpi=120, bbox_inches='tight',
                            facecolor='black')
                print(f"     Saved -> {out_path}")
            if show:
                plt.show()
            plt.close(fig)

    if len(images) > 1:
        print_summary(results_list)


if __name__ == '__main__':
    main(sys.argv)
