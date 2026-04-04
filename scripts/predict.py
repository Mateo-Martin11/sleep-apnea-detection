"""Generate submission predictions on the test set.

Usage:
    python scripts/predict.py
    python scripts/predict.py --config configs/config.yaml --checkpoint checkpoints/unet1d_v1_best.pt
    python scripts/predict.py --threshold 0.45
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_data, SleepApneaDataset
from src.data.preprocessing import normalize, clip_outliers, load_stats, instance_normalize
from src.models.factory import build_model
from src.training.trainer import Trainer
from src.training.losses import build_loss
from src.training.metrics import apply_postprocessing
from src.utils.submission import generate_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--threshold", type=float, help="Override classification threshold")
    parser.add_argument("--min_duration", type=int, default=9, help="Min event duration in seconds (post-processing)")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve checkpoint path (prefer _final if it exists, else _best)
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        final = Path(cfg["output"]["checkpoint_dir"]) / f"{cfg['output']['model_name']}_final.pt"
        best  = Path(cfg["output"]["checkpoint_dir"]) / f"{cfg['output']['model_name']}_best.pt"
        ckpt_path = str(final if final.exists() else best)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Load test data ─────────────────────────────────────────────────────────
    print("Loading test data...")
    signals, _, sample_ids, _ = load_data(
        cfg["data"]["test_h5"],
        cfg["data"]["test_ids_csv"],
    )
    print(f"  Signals: {signals.shape}")

    # ── Normalize ─────────────────────────────────────────────────────────────
    if cfg["data"].get("instance_norm", False):
        signals = clip_outliers(instance_normalize(signals))
    else:
        mean, std = load_stats(cfg["data"]["stats_path"])
        signals = clip_outliers(normalize(signals, mean, std))

    test_ds = SleepApneaDataset(signals)
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        name=cfg["model"]["name"],
        n_channels=cfg["model"]["n_channels"],
        base_filters=cfg["model"]["base_filters"],
        dropout=cfg["model"].get("dropout", 0.2),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # ── Predict ───────────────────────────────────────────────────────────────
    loss_fn = build_loss(cfg["loss"]["name"])  # needed for Trainer init
    trainer = Trainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters()),
        loss_fn=loss_fn,
        device=device,
    )
    trainer.best_threshold = ckpt.get("best_threshold", 0.5)

    threshold = args.threshold or trainer.best_threshold
    best_f1 = ckpt.get('best_f1')
    f1_str = f"{best_f1:.4f}" if best_f1 is not None else "N/A (all-subjects mode)"
    print(f"Using threshold: {threshold:.2f}  (best val F1 was {f1_str})")

    predictions, probs = trainer.predict(test_loader, threshold=threshold)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Positive rate avant post-processing: {predictions.mean():.3%}")

    # ── Post-processing ───────────────────────────────────────────────────────
    if args.min_duration > 1:
        predictions = apply_postprocessing(predictions, min_duration=args.min_duration)
        print(f"Positive rate après post-processing (min_dur={args.min_duration}s): {predictions.mean():.3%}")

    # ── Save submission ────────────────────────────────────────────────────────
    output_path = args.output or str(
        Path(cfg["output"]["submission_dir"])
        / f"submission_{cfg['output']['model_name']}_thr{threshold:.2f}_mindur{args.min_duration}.csv"
    )
    generate_submission(predictions, sample_ids, output_path)


if __name__ == "__main__":
    main()
