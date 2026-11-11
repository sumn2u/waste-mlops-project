# src/generate_features.py
"""
Generate image features for drift monitoring.

Extracts simple numerical features from training (reference) and
validation (production) images, then writes them to CSV files
for Evidently.

Handles:
- Reading images from two folders (reference and production)
- Extracting RGB means, brightness, dimensions, aspect ratio
- Writing two CSVs with identical column schemas
- Verifying schemas match before monitor.py runs

Usage:
    python src/generate_features.py
    python src/generate_features.py --reference-dir data/processed/images/train --production-dir data/processed/images/val
"""

import argparse
import sys
import pandas as pd
import cv2
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate image features for drift monitoring."
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="data/processed/images/train",
        help="Folder of baseline images (default: data/processed/images/train)",
    )
    parser.add_argument(
        "--production-dir",
        type=str,
        default="data/processed/images/val",
        help="Folder of production images (default: data/processed/images/val)",
    )
    parser.add_argument(
        "--reference-out",
        type=str,
        default="reference_features.csv",
        help="Output CSV for reference features (default: reference_features.csv)",
    )
    parser.add_argument(
        "--production-out",
        type=str,
        default="production_features.csv",
        help="Output CSV for production features (default: production_features.csv)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max images per folder (default: 500)",
    )
    return parser.parse_args()


def extract_features(image_path: Path):
    """
    Extract simple numerical features from one image.
    Returns a dict of features, or None if the image cannot be read.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    return {
        "mean_r": float(img[:, :, 2].mean()),
        "mean_g": float(img[:, :, 1].mean()),
        "mean_b": float(img[:, :, 0].mean()),
        "brightness": float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()),
        "width": int(img.shape[1]),
        "height": int(img.shape[0]),
        "aspect_ratio": float(img.shape[1] / img.shape[0]),
    }


def collect_images(folder: Path, max_samples: int):
    """
    Collect image paths from a folder, capped at max_samples.
    """
    if not folder.exists():
        print(f"ERROR: Folder not found: {folder}")
        return []

    images = []
    for path in folder.iterdir():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    images = images[:max_samples]
    print(f"  {folder} -> {len(images)} images")
    return images


def process_folder(images, output_csv: Path):
    """
    Extract features from each image and write to CSV.
    Returns (rows_written, skipped).
    """
    rows = []
    skipped = 0

    for img_path in images:
        feat = extract_features(img_path)
        if feat is None:
            skipped += 1
            continue
        rows.append(feat)

    if not rows:
        print(f"ERROR: All images failed to load for {output_csv}")
        return 0, skipped

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    return len(df), skipped


def verify_schemas(reference_csv: Path, production_csv: Path):
    """
    Verify both CSVs have identical columns.
    Returns True if schemas match.
    """
    ref_df = pd.read_csv(reference_csv)
    prod_df = pd.read_csv(production_csv)

    if list(ref_df.columns) != list(prod_df.columns):
        print("ERROR: Feature schemas differ.")
        print(f"  Reference:  {list(ref_df.columns)}")
        print(f"  Production: {list(prod_df.columns)}")
        return False

    print(f"Schemas match. Columns: {list(ref_df.columns)}")
    return True


def main():
    args = parse_args()

    reference_dir = Path(args.reference_dir)
    production_dir = Path(args.production_dir)
    reference_out = Path(args.reference_out)
    production_out = Path(args.production_out)

    print("=" * 60)
    print("Image Feature Generation")
    print("=" * 60)
    print(f"Reference dir:  {reference_dir}")
    print(f"Production dir: {production_dir}")
    print(f"Max samples:    {args.max_samples}")
    print("=" * 60)

    # Step 1: collect images
    print("\nScanning image folders...")
    ref_images = collect_images(reference_dir, args.max_samples)
    prod_images = collect_images(production_dir, args.max_samples)

    if not ref_images or not prod_images:
        print("ERROR: missing images. Check your --reference-dir and --production-dir.")
        sys.exit(1)

    # Step 2: extract reference features
    print("\nProcessing reference features...")
    n_ref, skipped_ref = process_folder(ref_images, reference_out)
    print(f"  [{reference_out.name}] {n_ref} rows, skipped {skipped_ref}")
    if n_ref == 0:
        sys.exit(1)

    # Step 3: extract production features
    print("\nProcessing production features...")
    n_prod, skipped_prod = process_folder(prod_images, production_out)
    print(f"  [{production_out.name}] {n_prod} rows, skipped {skipped_prod}")
    if n_prod == 0:
        sys.exit(1)

    # Step 4: verify schemas match
    print("\nVerifying feature schemas...")
    if not verify_schemas(reference_out, production_out):
        sys.exit(1)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Reference features:  {reference_out}")
    print(f"Production features: {production_out}")
    print("\nNext steps:")
    print("  1. python src/monitor.py")


if __name__ == "__main__":
    main()