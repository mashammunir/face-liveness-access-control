"""
scripts/evaluate_liveness.py

Evaluates a trained liveness model on the held-out test set
(data/test/live/, data/test/spoof/) and reports standard classification
metrics plus anti-spoofing-specific metrics.

METRICS EXPLAINED
------------------
Standard classification metrics (treating LIVE as the positive class):
  - Accuracy:  (TP + TN) / (TP + TN + FP + FN)
               Fraction of all predictions that were correct.
  - Precision: TP / (TP + FP)
               Of everything predicted LIVE, how much really was LIVE.
  - Recall:    TP / (TP + FN)
               Of everything that really was LIVE, how much was caught.
  - F1-score:  2 * (Precision * Recall) / (Precision + Recall)
               Harmonic mean of precision and recall.

Anti-spoofing-specific metrics (ISO/IEC 30107-3 style definitions):
  - APCER (Attack Presentation Classification Error Rate):
        APCER = FP_spoof / N_spoof
        Fraction of SPOOF (attack) samples that were wrongly
        classified as LIVE. This is the security-critical number —
        it measures how often an attacker gets through.
  - BPCER (Bona Fide Presentation Classification Error Rate):
        BPCER = FN_live / N_live
        Fraction of genuine LIVE samples that were wrongly classified
        as SPOOF. This measures how often a real, legitimate user is
        incorrectly rejected.
  - ACER (Average Classification Error Rate):
        ACER = (APCER + BPCER) / 2
        A single combined summary number.

Where, for this task:
  N_spoof   = total number of true SPOOF test samples
  N_live    = total number of true LIVE test samples
  FP_spoof  = SPOOF samples predicted as LIVE (a successful "spoof")
  FN_live   = LIVE samples predicted as SPOOF (a false rejection)

These numbers are only ever reported from an actual run on
data/test/ — this script never fabricates or assumes performance
numbers.

USAGE
-----
    python scripts/evaluate_liveness.py --data-dir data --model-path models/liveness_model.pth

Outputs:
    results/confusion_matrix.png
    results/metrics.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.liveness import build_model, get_eval_transform, CLASS_NAMES
from scripts.train_liveness import LivenessImageDataset


def compute_anti_spoofing_metrics(y_true, y_pred, live_label=1, spoof_label=0):
    """
    Compute APCER, BPCER, ACER given true/predicted integer labels
    where live_label / spoof_label identify the two classes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    is_spoof = y_true == spoof_label
    is_live = y_true == live_label

    n_spoof = int(is_spoof.sum())
    n_live = int(is_live.sum())

    # FP_spoof: true spoof, predicted live (attack got through)
    fp_spoof = int(((y_true == spoof_label) & (y_pred == live_label)).sum())
    # FN_live: true live, predicted spoof (genuine user rejected)
    fn_live = int(((y_true == live_label) & (y_pred == spoof_label)).sum())

    apcer = fp_spoof / n_spoof if n_spoof > 0 else float("nan")
    bpcer = fn_live / n_live if n_live > 0 else float("nan")
    if n_spoof > 0 and n_live > 0:
        acer = (apcer + bpcer) / 2
    else:
        acer = float("nan")

    return {
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "n_spoof_samples": n_spoof,
        "n_live_samples": n_live,
        "spoof_classified_as_live": fp_spoof,
        "live_classified_as_spoof": fn_live,
    }


def plot_confusion_matrix(cm, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Liveness Detection — Confusion Matrix")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--split", default="test", choices=["test", "val", "train"],
                         help="Which split to evaluate on (default: test)")
    parser.add_argument("--model-path", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--architecture", choices=["mobilenet_v3_small", "resnet18"],
                         default="mobilenet_v3_small")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args()

    eval_dir = args.data_dir / args.split
    if not eval_dir.exists():
        print(f"ERROR: {eval_dir} does not exist. "
              f"Run scripts/preprocess_dataset.py first.")
        sys.exit(1)

    if not args.model_path.exists():
        print(f"ERROR: no trained model found at {args.model_path}. "
              f"Run scripts/train_liveness.py first — evaluation requires "
              f"an actual trained model; this script will not fabricate "
              f"or guess performance numbers.")
        sys.exit(1)

    dataset = LivenessImageDataset(eval_dir, transform=get_eval_transform())
    if len(dataset) == 0:
        print(f"ERROR: no images found under {eval_dir}.")
        sys.exit(1)

    print(f"Evaluating on {args.split} split: {len(dataset)} samples  {dataset.class_counts()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.architecture, pretrained=False)
    try:
        state = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    except Exception as e:
        print(f"ERROR: failed to load model weights from {args.model_path}: {e}")
        sys.exit(1)
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    all_true, all_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_pred.extend(preds.tolist())
            all_true.extend(labels.numpy().tolist())

    live_idx = CLASS_NAMES.index("live")   # positive class = 1
    spoof_idx = CLASS_NAMES.index("spoof")  # negative class = 0

    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, pos_label=live_idx, zero_division=0)
    recall = recall_score(all_true, all_pred, pos_label=live_idx, zero_division=0)
    f1 = f1_score(all_true, all_pred, pos_label=live_idx, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=[spoof_idx, live_idx])

    spoof_metrics = compute_anti_spoofing_metrics(
        all_true, all_pred, live_label=live_idx, spoof_label=spoof_idx
    )

    print("\n--- Standard classification metrics ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\n--- Anti-spoofing metrics ---")
    print(f"APCER (attacks that fooled the system): {spoof_metrics['APCER']:.4f}")
    print(f"BPCER (real users wrongly rejected):    {spoof_metrics['BPCER']:.4f}")
    print(f"ACER  (average of the two):             {spoof_metrics['ACER']:.4f}")

    print("\nConfusion matrix (rows=true, cols=predicted), order=[spoof, live]:")
    print(cm)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    cm_path = args.results_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, ["spoof", "live"], cm_path)
    print(f"\nConfusion matrix image saved to: {cm_path}")

    metrics_out = {
        "split": args.split,
        "n_samples": len(dataset),
        "class_counts": dataset.class_counts(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["spoof", "live"],
        **spoof_metrics,
    }
    metrics_path = args.results_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
