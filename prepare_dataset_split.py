"""
Prepare a clean train/validation/test split from the original Brain Tumor MRI dataset
and materialize it as a folder layout under `data/`.

Expected original layout (downloaded dataset, read-only):

  Dataset/
    Training/
      Tumor/
      No Tumor/
    Testing/
      Tumor/
      No Tumor/

What this script does:
  - Loads all image paths and labels from `Dataset/Training` and `Dataset/Testing`.
  - Splits the Training set into:
      * train: (1 - val_fraction)
      * validation: val_fraction
    using a stratified split so class proportions are preserved.
  - Uses the entire Testing set as the held-out test set.
  - Creates a folder layout under `data/` (use --dry-run to skip creation):

      data/
        train/
          tumor/
          no_tumor/
        val/
          tumor/
          no_tumor/
        test/
          tumor/
          no_tumor/

    by symlinking from `Dataset/` (no data duplication). Use `--copy` to copy files instead.

Typical usage (from the project root):

  python prepare_dataset_split.py --dataset-dir Dataset

This creates `data/train`, `data/val`, and `data/test`, which the training script
(`train_monai_resnet.py`) will use automatically. Use --dry-run to only print counts.

Key options:

  --dataset-dir      Root of the original dataset (default: Dataset)
  --data-dir         Where to create train/val/test (default: data)
  --dry-run          Only print split counts; do not create folders
  --copy             Copy files instead of symlinking (uses more disk space)
  --val-fraction     Fraction of Training to use as validation (default: 0.15)
  --seed             Random seed for reproducibility (default: 42)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
CLASS_DIRS = ("tumor", "no_tumor")  # 1 -> tumor, 0 -> no_tumor


def normalize_class(name: str) -> int | None:
    """Map folder name to label: 0 = no_tumor, 1 = tumor."""
    n = name.strip().lower().replace(" ", "_")
    if n == "tumor":
        return 1
    if n in ("no_tumor", "notumor"):
        return 0
    return None


def collect_paths_and_labels(root: Path, subdir: str) -> Tuple[List[str], List[int]]:
    """
    Collect image paths and labels from root/subdir, which must contain
    class folders (e.g. Tumor, No Tumor).
    """
    base = root / subdir
    if not base.is_dir():
        return [], []

    paths: List[str] = []
    labels: List[int] = []

    for class_dir in sorted(base.iterdir()):
        if not class_dir.is_dir():
            continue
        label = normalize_class(class_dir.name)
        if label is None:
            continue
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(os.path.normpath(str(f.resolve())))
                labels.append(label)

    return paths, labels


def populate_split_dir(
    out_root: Path,
    split_name: str,
    paths: List[str],
    labels: List[int],
    use_copy: bool,
) -> None:
    """Create out_root/split_name/tumor and no_tumor and symlink/copy files into them."""
    for class_dir in CLASS_DIRS:
        (out_root / split_name / class_dir).mkdir(parents=True, exist_ok=True)
    used: dict[str, set[str]] = {"tumor": set(), "no_tumor": set()}
    for path, label in zip(paths, labels):
        class_dir = CLASS_DIRS[label]
        dest_dir = out_root / split_name / class_dir
        src = Path(path)
        name = src.name
        stem, suffix = src.stem, src.suffix
        dest_name = name
        idx = 1
        while dest_name in used[class_dir]:
            dest_name = f"{stem}_{idx}{suffix}"
            idx += 1
        used[class_dir].add(dest_name)
        dest_path = dest_dir / dest_name
        if use_copy:
            shutil.copy2(src, dest_path)
        else:
            try:
                dest_path.symlink_to(src.resolve())
            except OSError:
                # e.g. Windows without symlink privilege; fall back to copy
                shutil.copy2(src, dest_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split original Brain Tumor dataset into train/val/test."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="Dataset",
        help="Root directory containing Training/ and Testing/ (default: Dataset)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of Training set to use as validation (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split (default: 42)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory under which to create train/val/test (default: data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print split counts; do not create data/train, data/val, data/test",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of symlinking (use if symlinks fail or you want a standalone data/)",
    )
    args = parser.parse_args()

    root = Path(args.dataset_dir)
    if not root.is_dir():
        raise SystemExit(f"Dataset directory not found: {root.absolute()}")

    # Load Training and Testing
    train_all_paths, train_all_labels = collect_paths_and_labels(root, "Training")
    test_paths, test_labels = collect_paths_and_labels(root, "Testing")

    if not train_all_paths:
        raise SystemExit(
            f"No images found under {root / 'Training'}. "
            "Expected subfolders 'Tumor' and 'No Tumor' with image files."
        )
    if not test_paths:
        raise SystemExit(
            f"No images found under {root / 'Testing'}. "
            "Expected subfolders 'Tumor' and 'No Tumor' with image files."
        )

    # Stratified split of Training into train + val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_all_paths,
        train_all_labels,
        test_size=args.val_fraction,
        stratify=train_all_labels,
        random_state=args.seed,
    )

    data_dir = Path(args.data_dir)
    if args.dry_run:
        print(f"  Train:      {len(train_paths):,} images")
        print(f"  Validation: {len(val_paths):,} images")
        print(f"  Test:       {len(test_paths):,} images")
        print("\nRun without --dry-run to create data/train, data/val, data/test.")
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        populate_split_dir(
            data_dir, "train", train_paths, train_labels, use_copy=args.copy
        )
        populate_split_dir(
            data_dir, "val", val_paths, val_labels, use_copy=args.copy
        )
        populate_split_dir(
            data_dir, "test", test_paths, test_labels, use_copy=args.copy
        )
        print(f"Created folder layout under: {data_dir.absolute()}")
        print(f"  {data_dir / 'train'}/  ({len(train_paths):,} images)")
        print(f"  {data_dir / 'val'}/    ({len(val_paths):,} images)")
        print(f"  {data_dir / 'test'}/   ({len(test_paths):,} images)")
        print("  (symlinks)" if not args.copy else "  (copies)")
        print("\nTrain with: python train_monai_resnet.py")
        print("(Training script will use data/train, data/val, data/test automatically.)")


if __name__ == "__main__":
    main()
