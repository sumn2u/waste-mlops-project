"""
Upload the trained YOLO model directly to Hugging Face Hub.

This bypasses Git LFS/Xet issues by using the huggingface_hub API.

Usage:
    python src/upload_model.py
    python src/upload_model.py --repo iamsuman/waste-mlops --repo-type space
    python src/upload_model.py --repo iamsuman/waste-classifier-model --repo-type model
"""

import os
import sys
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload trained YOLO model to Hugging Face Hub."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="app/best.pt",
        help="Path to the model file (default: app/best.pt)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., iamsuman/waste-mlops)",
    )
    parser.add_argument(
        "--repo-type",
        type=str,
        default="space",
        choices=["space", "model", "dataset"],
        help="Repository type (default: space)",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default="best.pt",
        help="Destination filename in the repo (default: best.pt)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HF token (defaults to HF_TOKEN env var or cached login)",
    )
    return parser.parse_args()


def get_token(args) -> str:
    """Resolve the Hugging Face token from args, env, or cached login."""
    if args.token:
        return args.token

    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token

    # Fall back to cached token from `huggingface-cli login`
    try:
        from huggingface_hub import HfFolder
        cached = HfFolder.get_token()
        if cached:
            return cached
    except Exception:
        pass

    return None


def validate_model(model_path: Path) -> bool:
    """Verify the model file exists and has a reasonable size."""
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("Run training and copy first:")
        print("  python src/train.py")
        print("  python src/copy_model.py")
        return False

    if not model_path.is_file():
        print(f"ERROR: Not a file: {model_path}")
        return False

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model: {model_path}")
    print(f"  Size: {size_mb:.2f} MB")
    return True


def upload(model_path: Path, repo_id: str, repo_type: str,
           path_in_repo: str, token: str) -> bool:
    """Upload the model file to Hugging Face Hub."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print(f"\nUploading to {repo_type}: {repo_id}")
    print(f"  Destination: {path_in_repo}")

    try:
        url = api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"\nUpload successful!")
        print(f"  URL: {url}")
        return True

    except Exception as e:
        print(f"\nERROR: Upload failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your HF token has write access")
        print("  2. Verify the repository exists")
        print("  3. Ensure you have permission to push to this repo")
        return False


def main():
    args = parse_args()

    print("=" * 60)
    print("Upload Model to Hugging Face Hub")
    print("=" * 60)
    print(f"Model:     {args.model}")
    print(f"Repo:      {args.repo}")
    print(f"Repo type: {args.repo_type}")
    print("=" * 60)

    # ============================================================
    # STEP 1: Validate model file
    # ============================================================
    model_path = Path(args.model)
    if not validate_model(model_path):
        sys.exit(1)

    # ============================================================
    # STEP 2: Resolve token
    # ============================================================
    token = get_token(args)
    if not token:
        print("\nERROR: No Hugging Face token found.")
        print("\nOptions:")
        print("  1. Run: huggingface-cli login")
        print("  2. Set env: export HF_TOKEN=hf_xxxxx")
        print("  3. Pass: --token hf_xxxxx")
        print("\nGenerate a token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    print(f"Token: {'*' * 10}{token[-4:]}")

    # ============================================================
    # STEP 3: Upload
    # ============================================================
    success = upload(
        model_path=model_path,
        repo_id=args.repo,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
        token=token,
    )

    if not success:
        sys.exit(1)

    # ============================================================
    # STEP 4: Report
    # ============================================================
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Model uploaded to: {args.repo}")
    print(f"\nView your space: https://huggingface.co/spaces/{args.repo}")


if __name__ == "__main__":
    main()