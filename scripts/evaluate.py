"""Evaluate a checkpoint on the validation set (or any labeled split).

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/unet1d_v1_best.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_data, SleepApneaDataset, subject_train_val_split
from src.data.preprocessing import normalize, clip_outliers, load_stats
from src.models.factory import build_model
from src.training.losses import build_loss
from src.training.metrics import compute_f1_events, threshold_predictions, find_best_threshold
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", help="Path to checkpoint .pt file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or str(
        Path(cfg["output"]["checkpoint_dir"]) / f"{cfg['output']['model_name']}_best.pt"
    )
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Load + split data
    signals, labels, sample_ids, subject_ids = load_data(
        cfg["data"]["train_h5"],
        cfg["data"]["train_ids_csv"],
        cfg["data"]["train_labels_csv"],
    )
    _, _, _, X_val, y_val, _ = subject_train_val_split(
        signals, labels, subject_ids,
        val_ratio=cfg["data"]["val_ratio"],
        seed=cfg["data"]["seed"],
    )

    mean, std = load_stats(cfg["data"]["stats_path"])
    X_val = clip_outliers(normalize(X_val, mean, std))

    val_ds = SleepApneaDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    model = build_model(
        name=cfg["model"]["name"],
        n_channels=cfg["model"]["n_channels"],
        base_filters=cfg["model"]["base_filters"],
        dropout=cfg["model"].get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Collect probabilities
    all_probs = []
    with torch.no_grad():
        for x, _ in val_loader:
            logits = model(x.to(device))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(all_probs)

    # Evaluate at multiple thresholds
    print("\nThreshold sweep:")
    print(f"{'Threshold':>12} {'F1':>8} {'Precision':>12} {'Recall':>10}")
    for thr in np.arange(0.2, 0.8, 0.05):
        y_pred = threshold_predictions(probs, thr)
        f1, p, r = compute_f1_events(y_val, y_pred)
        marker = "  ← best" if abs(thr - ckpt.get("best_threshold", 0.5)) < 0.026 else ""
        print(f"{thr:>12.2f} {f1:>8.4f} {p:>12.4f} {r:>10.4f}{marker}")


if __name__ == "__main__":
    main()
