"""
scripts/train_liveness.py

Trains the face liveness (LIVE vs SPOOF) classifier on the preprocessed
dataset produced by scripts/preprocess_dataset.py.

Expects the standard folder structure:

    data/
        train/
            live/
            spoof/
        val/
            live/
            spoof/

Saves the best checkpoint (by validation accuracy) to
models/liveness_model.pth, plus a training_history.json with per-epoch
metrics under results/.

USAGE
-----
    python scripts/train_liveness.py \\
        --data-dir data \\
        --epochs 15 --batch-size 32 --lr 1e-4 \\
        --architecture mobilenet_v3_small

Defaults are read from config.py so this also works with no arguments
at all: `python scripts/train_liveness.py`
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.liveness import build_model, get_train_transform, get_eval_transform, CLASS_NAMES


class LivenessImageDataset(Dataset):
    """
    Simple dataset that reads all images under
        <root>/live/*.{jpg,png,...}
        <root>/spoof/*.{jpg,png,...}
    and labels them accordingly (live=1, spoof=0), matching CLASS_NAMES.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # list[(path, label_int)]

        for class_name in CLASS_NAMES:  # ["spoof", "live"] -> label 0, 1
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue
            label = CLASS_NAMES.index(class_name)
            for f in sorted(class_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Loaded as a numpy array (not PIL) so this matches exactly what
        # src.liveness.LivenessDetector.predict() feeds through the same
        # transform pipeline at inference time (ToPILImage expects an
        # ndarray or tensor as its first input).
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            image = self.transform(image)
        return image, label

    def class_counts(self):
        counts = {name: 0 for name in CLASS_NAMES}
        for _, label in self.samples:
            counts[CLASS_NAMES[label]] += 1
        return counts


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.TRAIN_LEARNING_RATE)
    parser.add_argument("--architecture", choices=["mobilenet_v3_small", "resnet18"],
                         default="mobilenet_v3_small")
    parser.add_argument("--patience", type=int, default=config.EARLY_STOPPING_PATIENCE,
                         help="Stop if val accuracy doesn't improve for this many epochs")
    parser.add_argument("--model-out", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--no-pretrained", action="store_true",
                         help="Train from random init instead of ImageNet-pretrained weights")
    args = parser.parse_args()

    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        print(f"ERROR: expected {train_dir} and {val_dir} to exist. "
              f"Run scripts/preprocess_dataset.py first.")
        sys.exit(1)

    train_ds = LivenessImageDataset(train_dir, transform=get_train_transform())
    val_ds = LivenessImageDataset(val_dir, transform=get_eval_transform())

    if len(train_ds) == 0:
        print(f"ERROR: no training images found under {train_dir}. "
              f"Did scripts/preprocess_dataset.py run successfully?")
        sys.exit(1)
    if len(val_ds) == 0:
        print(f"ERROR: no validation images found under {val_dir}. "
              f"Did scripts/preprocess_dataset.py run successfully?")
        sys.exit(1)

    print(f"Train samples: {len(train_ds)}  {train_ds.class_counts()}")
    print(f"Val samples:   {len(val_ds)}  {val_ds.class_counts()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(args.architecture, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0
    epochs_without_improvement = 0
    history = []

    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for up to {args.epochs} epochs "
          f"(early stopping patience={args.patience})...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.model_out)
            print(f"  -> New best val_acc={val_acc:.4f}. Saved to {args.model_out}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: no improvement for {args.patience} epochs.")
                break

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    history_path = config.RESULTS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump({
            "architecture": args.architecture,
            "best_val_accuracy": best_val_acc,
            "epochs_run": len(history),
            "history": history,
        }, f, indent=2)

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    print(f"Best model saved to: {args.model_out}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    main()
