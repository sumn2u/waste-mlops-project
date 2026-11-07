# src/prepare_data.py
"""
Reorganize the Kaggle Garbage Classification v2 BBox dataset
into the standard YOLO train/val structure.

Handles:
- Per-class folder layout (biological/, cardboard/, ...)
- Class ID remapping (if labels all use 0, they get the correct global index)
- 80/20 train/val split
- Coordinate validation (must be normalized 0-1)

Usage:
    python src/prepare_data.py
    python src/prepare_data.py --raw data --out data/processed --val-split 0.2
"""

import argparse
import random
import shutil
import sys
from pathlib import Path


# Class order MUST match src/data.yaml
CLASSES = [
    "biological",
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reorganize Kaggle dataset into YOLO train/val structure."
    )
    parser.add_argument(
        "--raw",
        type=str,
        default="data",
        help="Path to the unzipped Kaggle dataset root (default: data)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed",
        help="Output directory for YOLO-ready dataset (default: data/processed)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (default: move)",
    )
    return parser.parse_args()


def create_output_dirs(out_root: Path):
    """Create the YOLO directory structure."""
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    print(f"Created output structure at: {out_root}")


def collect_pairs(raw_root: Path):
    """
    Walk the per-class folders and collect (image, label, class_idx) triples.
    """
    pairs = []
    missing = []

    for class_idx, class_name in enumerate(CLASSES):
        img_dir = raw_root / class_name / "images"
        lbl_dir = raw_root / class_name / "labels"

        if not img_dir.exists():
            missing.append(str(img_dir))
            continue

        class_count = 0
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            pairs.append((img_path, label_path, class_idx))
            class_count += 1

        print(f"  {class_name:12s} -> {class_count:5d} pairs")

    if missing:
        print("\nWARNING: missing directories:")
        for m in missing:
            print(f"  - {m}")

    return pairs


def inspect_label_format(pairs, sample_size=5):
    """
    Inspect a few label files to determine if class IDs need remapping.
    Returns True if remapping appears necessary.
    """
    print("\nInspecting label format (sample):")
    print("-" * 60)

    seen_ids = set()
    sample = random.sample(pairs, min(sample_size, len(pairs)))

    for img, lbl, class_idx in sample:
        lines = lbl.read_text().strip().splitlines()
        for line in lines[:1]:
            parts = line.split()
            if len(parts) == 5:
                seen_ids.add(int(parts[0]))
                print(
                    f"  {lbl.parent.parent.name:12s} | "
                    f"class_idx={class_idx} | label class_id={parts[0]} | "
                    f"{lbl.name}"
                )

    print("-" * 60)
    if seen_ids == {0}:
        print("All sampled labels use class_id=0 -> REMAPPING REQUIRED")
        return True
    elif len(seen_ids) > 1:
        print(f"Labels already use global IDs: {sorted(seen_ids)} -> no remapping")
        return False
    else:
        print(f"Single class ID seen: {seen_ids} -> remapping will be applied")
        return True


def validate_label_line(parts):
    """Return (ok, error_message) for a single YOLO label line."""
    if len(parts) != 5:
        return False, f"expected 5 fields, got {len(parts)}"

    try:
        int(parts[0])
        coords = [float(x) for x in parts[1:5]]
    except ValueError:
        return False, "non-numeric value"

    for c in coords:
        if c < 0 or c > 1:
            return False, f"coord out of [0,1]: {c}"

    return True, None


def write_label(src_label: Path, dst_label: Path, new_class_id: int):
    """
    Read a YOLO label file, replace the class ID with new_class_id,
    validate coordinates, and write to dst_label.
    Returns (num_boxes, errors).
    """
    errors = []
    lines = src_label.read_text().strip().splitlines()
    new_lines = []

    for i, line in enumerate(lines, 1):
        parts = line.strip().split()
        ok, err = validate_label_line(parts)
        if not ok:
            errors.append(f"{src_label.name}:{i} - {err}")
            continue

        parts[0] = str(new_class_id)
        new_lines.append(" ".join(parts))

    if new_lines:
        dst_label.write_text("\n".join(new_lines))

    return len(new_lines), errors


def process_split(pairs, out_root: Path, split: str, remap: bool, copy: bool):
    """Process one split (train or val)."""
    n_images = 0
    n_boxes = 0
    all_errors = []

    move_fn = shutil.copy2 if copy else shutil.move

    for img_path, lbl_path, class_idx in pairs:
        dst_img = out_root / "images" / split / img_path.name
        dst_lbl = out_root / "labels" / split / lbl_path.name

        move_fn(str(img_path), str(dst_img))

        new_class_id = class_idx if remap else int(
            lbl_path.read_text().strip().split()[0]
        )
        n_boxes_this, errors = write_label(lbl_path, dst_lbl, new_class_id)

        n_images += 1
        n_boxes += n_boxes_this
        all_errors.extend(errors)

    print(f"  [{split}] {n_images} images, {n_boxes} boxes")
    return n_images, n_boxes, all_errors


def write_data_yaml(out_root: Path, data_yaml_path: Path):
    """Write the data.yaml file pointing to the processed dataset."""
    # Use a relative path from the yaml's location
    try:
        rel_path = out_root.resolve().relative_to(data_yaml_path.parent.resolve())
    except ValueError:
        rel_path = out_root.resolve()

    content = (
        f"# Auto-generated by prepare_data.py\n"
        f"path: {rel_path}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n"
    )
    for idx, name in enumerate(CLASSES):
        content += f"  {idx}: {name}\n"

    data_yaml_path.write_text(content)
    print(f"Wrote {data_yaml_path}")


def main():
    args = parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)

    print("=" * 60)
    print("YOLO Dataset Preparation")
    print("=" * 60)
    print(f"Raw data:   {raw_root}")
    print(f"Output:     {out_root}")
    print(f"Val split:  {args.val_split}")
    print(f"Seed:       {args.seed}")
    print(f"Mode:       {'copy' if args.copy else 'move'}")
    print("=" * 60)

    if not raw_root.exists():
        print(f"ERROR: raw data directory not found: {raw_root}")
        sys.exit(1)

    # Step 1: collect pairs
    print("\nScanning class folders...")
    pairs = collect_pairs(raw_root)

    if not pairs:
        print("ERROR: no image/label pairs found. Check your --raw path.")
        sys.exit(1)

    print(f"\nTotal pairs: {len(pairs)}")

    # Step 2: inspect format
    random.seed(args.seed)
    remap = inspect_label_format(pairs)

    # Step 3: shuffle and split
    random.shuffle(pairs)
    split_idx = int(len(pairs) * (1 - args.val_split))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]

    print(f"\nSplit: {len(train_pairs)} train / {len(val_pairs)} val")

    # Step 4: create output structure
    print("\nCreating output structure...")
    create_output_dirs(out_root)

    # Step 5: process each split
    print("\nProcessing splits...")
    _, _, train_errors = process_split(
        train_pairs, out_root, "train", remap, args.copy
    )
    _, _, val_errors = process_split(
        val_pairs, out_root, "val", remap, args.copy
    )

    # Step 6: report errors
    all_errors = train_errors + val_errors
    if all_errors:
        print(f"\nWARNING: {len(all_errors)} label issues found:")
        for err in all_errors[:20]:
            print(f"  - {err}")
        if len(all_errors) > 20:
            print(f"  ... and {len(all_errors) - 20} more")

    # Step 7: write data.yaml
    data_yaml_path = Path("src/data.yaml")
    data_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    write_data_yaml(out_root, data_yaml_path)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Processed dataset: {out_root}")
    print(f"Data config:       {data_yaml_path}")
    print("\nNext steps:")
    print("  1. python src/validate_data.py")
    print("  2. python src/train.py")


if __name__ == "__main__":
    main()