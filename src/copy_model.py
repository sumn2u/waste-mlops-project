"""
Copy the trained YOLO model into the app/ folder for Hugging Face deployment.

Usage:
    python src/copy_model.py
    python src/copy_model.py --source waste-classification/short-run/weights/best.pt
    python src/copy_model.py --source waste-classification/yolov8n-run/weights/best.pt --dest app/best.pt
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy trained YOLO model to the app folder for deployment."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="waste-classification/yolov8n-run/weights/best.pt",
        help="Path to the trained model (default: waste-classification/yolov8n-run/weights/best.pt)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="app/best.pt",
        help="Destination path in the app folder (default: app/best.pt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination without prompting",
    )
    return parser.parse_args()


def validate_source(source: Path) -> bool:
    """Verify the source model file exists and has a reasonable size."""
    if not source.exists():
        print(f"ERROR: Source model not found: {source}")
        print("Run training first: python src/train.py")
        return False

    if not source.is_file():
        print(f"ERROR: Source is not a file: {source}")
        return False

    size_mb = source.stat().st_size / (1024 * 1024)
    if size_mb < 1:
        print(f"WARNING: Model file is very small ({size_mb:.2f} MB)")
        print("This may indicate a failed or incomplete training run.")

    print(f"Source model: {source}")
    print(f"  Size: {size_mb:.2f} MB")
    return True


def validate_destination(dest: Path) -> bool:
    """Check the destination folder exists and whether a file already exists."""
    dest_dir = dest.parent

    if not dest_dir.exists():
        print(f"Destination folder does not exist: {dest_dir}")
        print("Creating it...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        return True

    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"Existing model at destination: {dest} ({size_mb:.2f} MB)")
        return False

    return True


def copy_model(source: Path, dest: Path, force: bool) -> bool:
    """Copy the model file from source to destination."""
    if dest.exists() and not force:
        response = input(f"Overwrite {dest}? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            print("Copy cancelled.")
            return False

    shutil.copy2(source, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"Copied to: {dest} ({size_mb:.2f} MB)")
    return True


def verify_copy(source: Path, dest: Path) -> bool:
    """Verify the copied file matches the source size."""
    if not dest.exists():
        print("ERROR: Destination file missing after copy")
        return False

    src_size = source.stat().st_size
    dst_size = dest.stat().st_size

    if src_size != dst_size:
        print(f"ERROR: Size mismatch — source: {src_size}, dest: {dst_size}")
        return False

    print("Copy verified: source and destination sizes match.")
    return True


def check_git_tracking(dest: Path) -> bool:
    """
    Check if the destination file is tracked by Git.
    If it is, warn the user because HF will reject the push.
    Returns True if the file is safely untracked, False if it's tracked.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(dest)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"\nWARNING: {dest} is currently tracked by Git.")
            print("Hugging Face will reject any push that contains this binary.")
            print("\nTo fix, run:")
            print(f"  git rm --cached {dest}")
            print(f'  git commit -m "Remove {dest.name} from Git tracking"')
            return False
        else:
            print(f"\n{dest} is not tracked by Git. Safe for HF deployment.")
            return True
    except FileNotFoundError:
        print("\nGit not found — skipping Git tracking check.")
        return True
    except Exception as e:
        print(f"\nCould not check Git tracking: {e}")
        return True


def main():
    args = parse_args()

    source = Path(args.source)
    dest = Path(args.dest)

    print("=" * 60)
    print("Copy Trained Model to App Folder")
    print("=" * 60)
    print(f"Source: {source}")
    print(f"Dest:   {dest}")
    print("=" * 60)

    # ============================================================
    # STEP 1: Validate source
    # ============================================================
    if not validate_source(source):
        sys.exit(1)

    # ============================================================
    # STEP 2: Check destination
    # ============================================================
    validate_destination(dest)

    # ============================================================
    # STEP 3: Copy
    # ============================================================
    if not copy_model(source, dest, args.force):
        sys.exit(1)

    # ============================================================
    # STEP 4: Verify copy
    # ============================================================
    if not verify_copy(source, dest):
        sys.exit(1)

    # ============================================================
    # STEP 5: Check Git tracking status
    # ============================================================
    check_git_tracking(dest)

    # ============================================================
    # STEP 6: Report next steps
    # ============================================================
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Model ready for deployment at: {dest}")
    print("\nNext steps (upload via Hugging Face API, NOT git):")
    print("  python3 src/upload_model.py \\")
    print("      --repo iamsuman/waste-mlops \\")
    print("      --repo-type space")
    print("\nNote: Do NOT 'git add best.pt' — Hugging Face rejects")
    print("      binary files pushed through Git.")


if __name__ == "__main__":
    main()