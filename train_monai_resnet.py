import os
import random
from glob import glob
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from monai.data import ImageDataset, DataLoader
from monai.networks.nets import resnet18
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    Lambda,
    RandFlip,
    Resize,
    ScaleIntensity,
    ToTensor,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_image_paths_and_labels(
    data_root: str,
) -> Tuple[List[str], List[int]]:
    """
    Expect a directory structure like:

        data_root/
          tumor/
          no_tumor/

    Returns image paths and integer labels:
        0 -> no_tumor
        1 -> tumor
    """
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    image_paths: List[str] = []

    for ext in exts:
        pattern = os.path.join(data_root, "*", ext)
        image_paths.extend(glob(pattern))

    if not image_paths:
        raise RuntimeError(
            f"No images found under '{data_root}'. "
            "Ensure it contains 'tumor/' and 'no_tumor/' subfolders "
            "with image files (png/jpg/jpeg/bmp)."
        )

    labels: List[int] = []
    for p in image_paths:
        class_name = os.path.basename(os.path.dirname(p)).lower()
        if class_name == "no_tumor":
            labels.append(0)
        elif class_name == "tumor":
            labels.append(1)
        else:
            raise RuntimeError(
                f"Unexpected class folder '{class_name}' for file '{p}'. "
                "Expected folder names: 'tumor' and 'no_tumor'."
            )

    return image_paths, labels


def create_transforms():
    train_transforms = Compose(
        [
            EnsureChannelFirst(),
            # Some images are grayscale (1 channel) and some are RGB (3 channels).
            # Convert everything to a single-channel image so batches have shape [1, H, W].
            Lambda(lambda x: x.mean(axis=0, keepdims=True) if x.shape[0] != 1 else x),
            ScaleIntensity(),
            Resize((224, 224)),
            RandFlip(prob=0.5, spatial_axis=0),
            ToTensor(),
        ]
    )

    eval_transforms = Compose(
        [
            EnsureChannelFirst(),
            Lambda(lambda x: x.mean(axis=0, keepdims=True) if x.shape[0] != 1 else x),
            ScaleIntensity(),
            Resize((224, 224)),
            ToTensor(),
        ]
    )

    return train_transforms, eval_transforms


def build_dataloaders(
    train_imgs: List[str],
    train_labels: List[int],
    val_imgs: List[str],
    val_labels: List[int],
    test_imgs: List[str],
    test_labels: List[int],
    batch_size: int = 8,
):
    train_transforms, eval_transforms = create_transforms()

    # MONAI's ImageDataset signature is (image_files, seg_files=None, labels=None, ...),
    # so labels must be passed as a keyword argument to avoid being treated as seg_files.
    train_ds = ImageDataset(train_imgs, labels=train_labels, transform=train_transforms)
    val_ds = ImageDataset(val_imgs, labels=val_labels, transform=eval_transforms)
    test_ds = ImageDataset(test_imgs, labels=test_labels, transform=eval_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def build_model(device: torch.device) -> nn.Module:
    model = resnet18(
        spatial_dims=2,
        n_input_channels=1,
        num_classes=2,
    )
    model.to(device)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    print(f"  [train] {num_batches} batches in this epoch")

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Lightweight progress print every 20 batches (or for very small datasets, every batch).
        if (batch_idx + 1) % max(1, num_batches // 20 or 1) == 0 or (batch_idx + 1) == num_batches:
            avg_so_far = running_loss / (batch_idx + 1)
            print(f"    batch {batch_idx + 1}/{num_batches} - running avg loss: {avg_so_far:.4f}")

    return running_loss / max(len(loader), 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, np.array(all_labels), np.array(all_preds)


def plot_sample_prediction(
    loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    save_path: str = "sample_prediction.png",
) -> None:
    model.eval()
    batch = next(iter(loader))
    x, y = batch

    with torch.no_grad():
        preds = model(x.to(device)).argmax(dim=1).cpu()

    plt.figure(figsize=(4, 4))
    plt.imshow(x[0][0], cmap="gray")
    plt.title(f"Pred: {preds[0].item()} | GT: {y[0].item()}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a MONAI ResNet-18 on Brain Tumor MRI (Tumor / No Tumor)."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root for folder layout: data-dir/train, data-dir/val, data-dir/test (each with tumor/, no_tumor/). Checked first.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/raw_data",
        help="Fallback: single directory with 'tumor/' and 'no_tumor/' when folder layout (data/train, val, test) is not found.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save model checkpoints and figures.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 2: get train/val/test paths and labels (folder layout or data-root with 70/15/15 split)
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")
    use_folders = (
        os.path.isdir(train_dir)
        and os.path.isdir(val_dir)
        and os.path.isdir(test_dir)
    )
    if use_folders:
        try:
            train_imgs, train_labels = collect_image_paths_and_labels(train_dir)
            val_imgs, val_labels = collect_image_paths_and_labels(val_dir)
            test_imgs, test_labels = collect_image_paths_and_labels(test_dir)
            if train_imgs and val_imgs and test_imgs:
                print(f"Using folder layout: {args.data_dir}/train, {args.data_dir}/val, {args.data_dir}/test")
            else:
                use_folders = False
        except RuntimeError:
            use_folders = False
    if not use_folders:
        image_paths, labels = collect_image_paths_and_labels(args.data_root)
        print(f"Folder layout not found; using --data-root and 70/15/15 split. Total images: {len(image_paths)}")
        train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
            image_paths,
            labels,
            test_size=0.30,
            stratify=labels,
            random_state=args.seed,
        )
        val_imgs, test_imgs, val_labels, test_labels = train_test_split(
            temp_imgs,
            temp_labels,
            test_size=0.50,
            stratify=temp_labels,
            random_state=args.seed,
        )

    print(f"Train size: {len(train_imgs)}")
    print(f"Val size:   {len(val_imgs)}")
    print(f"Test size:  {len(test_imgs)}")

    # Step 3 & 4: transforms, datasets, dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        train_imgs,
        train_labels,
        val_imgs,
        val_labels,
        test_imgs,
        test_labels,
        batch_size=args.batch_size,
    )

    # Step 5: model
    model = build_model(device)

    # Step 6: loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Step 7 & 8: training with validation monitoring
    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "best_resnet18.pth")

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_acc, _, _ = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved to {best_model_path}")

    print(f"Best validation accuracy: {best_val_acc:.2%}")

    # Load best model before final evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path} for final evaluation.")

    # Step 9: final evaluation on the test set
    test_acc, test_labels_np, test_preds_np = evaluate(model, test_loader, device)
    print(f"\nTest Accuracy: {test_acc:.2%}")

    print("\nClassification report (Test set):")
    print(
        classification_report(
            test_labels_np,
            test_preds_np,
            target_names=["No Tumor", "Tumor"],
        )
    )

    print("Confusion matrix (Test set):")
    print(confusion_matrix(test_labels_np, test_preds_np))

    # Step 10: optional visualization of predictions
    sample_pred_path = os.path.join(args.output_dir, "sample_prediction.png")
    plot_sample_prediction(
        loader=test_loader,
        model=model,
        device=device,
        save_path=sample_pred_path,
    )
    print(f"Sample prediction image saved to: {sample_pred_path}")


if __name__ == "__main__":
    main()

