import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from plantcv import plantcv as pcv


def set_backend():
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


BACKEND = set_backend()
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
    del os.environ['QT_QPA_PLATFORM']
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff',
    '.tif', '.JPG', '.JPEG', '.PNG'
}


def compute_transformations(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image: {image_path}")
        exit()

    results = {}
    results['original'] = img

    pcv.params.debug = None
    gray_s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')
    gaussian = pcv.gaussian_blur(
        img=gray_s, ksize=(11, 11), sigma_x=0, sigma_y=None)
    results['gaussian'] = gaussian

    binary = pcv.threshold.binary(
        gray_img=gray_s, threshold=50, object_type='light')
    mask = pcv.fill(bin_img=binary, size=200)
    results['mask'] = mask

    roi = img.copy()
    overlay = np.zeros_like(img)
    overlay[mask > 0] = [0, 255, 0]
    roi = cv2.addWeighted(roi, 0.5, overlay, 0.5, 0)
    h, w = roi.shape[:2]
    cv2.rectangle(roi, (5, 5), (w - 5, h - 5), (255, 0, 0), 4)
    results['roi'] = roi

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    analyze = img.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(analyze, [largest], -1, (255, 0, 255), 3)
        if len(largest) >= 5:
            cv2.ellipse(analyze, cv2.fitEllipse(largest), (255, 128, 0), 2)
        x, y, bw, bh = cv2.boundingRect(largest)
        cv2.rectangle(analyze, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < cv2.contourArea(largest) * 0.1:
                cv2.drawContours(analyze, [cnt], -1, (0, 0, 255), 2)
    results['analyze'] = analyze

    landmarks = img.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest)
        indices = np.linspace(0, len(largest) - 1, 20, dtype=int)
        for idx in indices:
            cv2.circle(landmarks, tuple(largest[idx][0]), 5, (255, 0, 0), -1)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < largest_area * 0.05:
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(landmarks, (cx, cy), 6, (0, 128, 255), -1)
        M_leaf = cv2.moments(largest)
        if M_leaf['m00'] != 0:
            cx_leaf = int(M_leaf['m10'] / M_leaf['m00'])
            x, y, bw, bh = cv2.boundingRect(largest)
            for i in range(5):
                cv2.circle(
                    landmarks,
                    (cx_leaf, y + int(bh * i / 4)),
                    5, (0, 255, 0), -1
                )
    results['landmarks'] = landmarks

    return results


def build_color_histogram(img, mask):
    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    channels = {
        'blue':          (b,            '#1f77b4'),
        'green':         (g,            '#2ca02c'),
        'red':           (r,            '#d62728'),
        'hue':           (hsv[:, :, 0], '#9467bd'),
        'saturation':    (hsv[:, :, 1], '#17becf'),
        'value':         (hsv[:, :, 2], '#bcbd22'),
        'lightness':     (lab[:, :, 0], '#7f7f7f'),
        'green-magenta': (lab[:, :, 1], '#e377c2'),
        'blue-yellow':   (lab[:, :, 2], '#ff7f0e'),
    }

    if mask is not None:
        total = np.count_nonzero(mask)
    else:
        total = img.shape[0] * img.shape[1]

    histograms = {}
    for name, (channel, color) in channels.items():
        if mask is not None:
            pixels = channel[mask > 0]
        else:
            pixels = channel.flatten()
        hist, edges = np.histogram(pixels, bins=256, range=(0, 256))
        x = (edges[:-1] + edges[1:]) / 2
        histograms[name] = (x, (hist / total) * 100, color)
    return histograms


def draw_figure(fig, results):
    img = results['original']
    gaussian = results['gaussian']
    mask = results['mask']
    roi = results['roi']
    analyze = results['analyze']
    landmarks = results['landmarks']

    gs = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.30)

    def show(ax, data, title, cmap=None):
        if cmap:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(cv2.cvtColor(data, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9)

    show(fig.add_subplot(gs[0, 0]), img, "Figure IV.1: Original")
    show(fig.add_subplot(gs[0, 1]), gaussian,
         "Figure IV.2: Gaussian blur", cmap='gray')
    show(fig.add_subplot(gs[0, 2]), mask, "Figure IV.3: Mask", cmap='gray')
    show(fig.add_subplot(gs[0, 3]), roi, "Figure IV.4: Roi objects")
    show(fig.add_subplot(gs[1, 0]), analyze, "Figure IV.5: Analyze object")
    show(fig.add_subplot(gs[1, 1]), landmarks, "Figure IV.6: Pseudolandmarks")

    ax_hist = fig.add_subplot(gs[1, 2:])
    for name, (x, y, color) in build_color_histogram(img, mask).items():
        ax_hist.plot(x, y, color=color, label=name, linewidth=1.2)
    ax_hist.set_title("Figure IV.7: Color histogram", fontsize=9)
    ax_hist.set_xlabel("Pixel intensity")
    ax_hist.set_ylabel("Proportion of pixels (%)")
    ax_hist.legend(title="color Channel", fontsize=7, title_fontsize=8,
                   loc='upper right', framealpha=0.7)
    ax_hist.set_xlim(0, 256)
    ax_hist.grid(True, alpha=0.3)


def display_transformations(image_path):
    results = compute_transformations(image_path)
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Leaf Transformations - {os.path.basename(image_path)}",
        fontsize=14, fontweight='bold'
    )
    draw_figure(fig, results)
    plt.show()


def save_transformations(image_path, dst_dir):
    results = compute_transformations(image_path)

    fig = plt.figure(figsize=(18, 12))
    basename = os.path.splitext(os.path.basename(image_path))[0]
    fig.suptitle(
        f"Leaf Transformations - {basename}",
        fontsize=13, fontweight='bold'
    )
    draw_figure(fig, results)

    os.makedirs(dst_dir, exist_ok=True)
    out_path = os.path.join(dst_dir, basename + "_transformations.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Saved -> {out_path}")
    return out_path


def parse_args(args):
    src = None
    dst = None

    i = 1
    while i < len(args):
        if args[i] in ('-src', '--src'):
            if i + 1 >= len(args):
                print("Error: -src requires a path.")
                exit()
            src = args[i + 1]
            i += 2
        elif args[i] in ('-dst', '--dst'):
            if i + 1 >= len(args):
                print("Error: -dst requires a directory path.")
                exit()
            dst = args[i + 1]
            i += 2
        else:
            src = args[i]
            i += 1

    return src, dst


def main(args):
    if len(args) < 2:
        print("Error: Use -src <path> or pass the path directly.")
        exit()

    src, dst = parse_args(args)

    if not src:
        print("Error: Use -src <path> or pass the path directly.")
        exit()

    if not os.path.exists(src):
        print(f"Error: path not found: {src}")
        exit()

    if os.path.isdir(src):
        if not dst:
            print("Error: -dst <directory>"
                  " is required when -src is a directory.")
            exit()

        image_files = []
        for f in os.listdir(src):
            if os.path.splitext(f)[1] in IMAGE_EXTENSIONS:
                image_files.append(f)

        if not image_files:
            print(f"No images found in: {src}")
            exit()

        print(f"Found {len(image_files)} image(s)"
              " in '{src}' -> saving to '{dst}'")
        for fname in sorted(image_files):
            print(f"  Processing: {fname}")
            try:
                save_transformations(
                    os.path.join(src, fname), dst)
            except Exception as e:
                print(f"  Warning - skipped {fname}: {e}")

        print(f"\nDone. All results saved in: {dst}")

    else:
        print(f"Processing: {src}")
        display_transformations(src)


if __name__ == "__main__":
    main(sys.argv)
