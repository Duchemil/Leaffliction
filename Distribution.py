import os
import sys
import matplotlib.pyplot as plt

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".tiff", ".JPG", ".PNG"
}


def count_images(root_dir):
    counts = {}
    for dirpath, dirs, filenames in os.walk(root_dir):
        if dirpath == root_dir:
            continue
        imgs = []
        for f in filenames:
            if os.path.splitext(f)[1] in IMAGE_EXTENSIONS:
                imgs.append(f)
        if imgs:
            label = os.path.relpath(dirpath, root_dir)
            counts[label] = len(imgs)
    return counts


def plot_distribution(counts, root_dir):
    labels = list(counts.keys())
    values = list(counts.values())

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Image distribution - {os.path.basename(root_dir)}",
        fontsize=14
    )

    ax_pie.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    ax_pie.set_title("Proportion per class")

    ax_bar.bar(labels, values)
    ax_bar.set_xlabel("Disease")
    ax_bar.set_ylabel("Number of images")
    ax_bar.set_title("Count per disease")
    plt.setp(ax_bar.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    plt.show()


def main(args):
    if len(args) != 2:
        print(f"Usage: python {args[0]} <directory>")
        exit()

    root_dir = args[1]
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        exit()

    counts = count_images(root_dir)
    if not counts:
        print("No images found in subdirectories.")
        exit()

    plot_distribution(counts, root_dir)


if __name__ == "__main__":
    main(sys.argv)
