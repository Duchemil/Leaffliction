import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # headless/back-end safe


IMAGE_EXTS_DEFAULT = {".jpg", ".jpeg", ".png", ".bmp",
                      ".gif", ".tif", ".tiff", ".webp"}


def is_image(path: Path, exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def scan_images(root: Path, exts: set[str]):
    """
    Returns:
      plant_counts: Counter of images per plant (first-level subdir).
      plant_to_subcats: dict[plant] -> Counter of subcategories under plant.
      plant_to_files: dict[plant] -> list[Path] of image files.
    """
    plant_counts = Counter()
    plant_to_subcats: dict[str, Counter] = defaultdict(Counter)
    plant_to_files: dict[str, list[Path]] = defaultdict(list)

    for p in root.rglob("*"):
        if not is_image(p, exts):
            continue
        rel = p.relative_to(root)
        parts = rel.parts
        if len(parts) == 0:
            # Shouldn't happen for rglob("*"), but guard anyway
            continue
        plant = parts[0]
        subcat = "(root)"  # images directly under the plant folder
        if len(parts) >= 2:
            subcat = parts[1]

        plant_counts[plant] += 1
        plant_to_subcats[plant][subcat] += 1
        plant_to_files[plant].append(p)

    return plant_counts, plant_to_subcats, plant_to_files


def ensure_out_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_")
                   else "_" for c in name.strip())
    return safe or "unnamed"


def save_pie(counts: Counter, title: str, out_path: Path):
    if not counts:
        print("No data for pie chart, skipping:", out_path)
        return
    labels, sizes = zip(*counts.most_common())
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=140,
        counterclock=False,
        textprops={"fontsize": 10},
    )
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_bar(counts: Counter, title: str, out_path: Path):
    if not counts:
        print("No data for bar chart, skipping:", out_path)
        return
    items = counts.most_common()
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]

    # Dynamic size to keep labels readable
    width = max(8, min(16, 0.6 * len(labels) + 4))
    height = max(4, min(12, 0.5 * len(labels) + 3))
    fig, ax = plt.subplots(figsize=(width, height))
    bars = ax.bar(range(len(labels)), values, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Image count")
    ax.set_title(title)

    # Add colors to bars
    colors = plt.cm.tab20.colors  # Use a colormap with diverse colors
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])

    # Annotate counts on bars
    for rect, val in zip(bars, values):
        ax.annotate(
            f"{val}",
            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def build_extension_counts(files: list[Path]) -> Counter:
    c = Counter()
    for f in files:
        c[f.suffix.lower()] += 1
    return c


def main():
    parser = argparse.ArgumentParser(
        description="Plot image distribution for a dataset directory."
    )
    parser.add_argument(
        "root",
        type=str,
        help="Root directory containing plant subfolders (e.g., ./images).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Error: '{root}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    exts = IMAGE_EXTS_DEFAULT

    out_dir = Path("charts").resolve()
    ensure_out_dir(out_dir)

    plant_counts, plant_to_subcats, plant_to_files = scan_images(root, exts)

    if not plant_counts:
        print("No images found. Check the directory or extensions.")
        sys.exit(0)

    # Overall pie: images per plant
    save_pie(
        plant_counts,
        title=f"Image distribution per plant in '{root.name}'",
        out_path=out_dir / f"pie_overall_{sanitize_filename(root.name)}.png",
    )

    # Overall bar: same data as the pie (images per plant)
    save_bar(
        plant_counts,
        title=f"Image distribution per plant in '{root.name}'",
        out_path=out_dir / f"bar_overall_{sanitize_filename(root.name)}.png",
    )
    print(f"Charts saved to: {out_dir}")


if __name__ == "__main__":
    main()
