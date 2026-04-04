"""Main training script.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/config.yaml
    python scripts/train.py --model cnn1d --batch_size 64 --n_epochs 50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_data, SleepApneaDataset, subject_train_val_split
from src.data.preprocessing import compute_stats, normalize, clip_outliers, save_stats, instance_normalize
from src.data.augmentation import get_train_transform
from src.models.factory import build_model
from src.training.losses import build_loss
from src.training.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of YAML config."""
    if args.model:
        cfg["model"]["name"] = args.model
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.n_epochs:
        cfg["training"]["n_epochs"] = args.n_epochs
    if args.lr:
        cfg["training"]["lr"] = args.lr
    if args.no_augment:
        cfg["training"]["augment"] = False
    return cfg


def build_scheduler(optimizer, cfg: dict):
    name = cfg["scheduler"].get("name")
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=cfg["scheduler"]["patience"],
            factor=cfg["scheduler"]["factor"],
            min_lr=cfg["scheduler"]["min_lr"],
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["n_epochs"],
        )
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", choices=["cnn1d", "unet1d"])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--all_subjects", action="store_true",
                        help="Train on all 22 subjects (no val split) for fixed n_epochs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = override_config(cfg, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data...")
    signals, labels, sample_ids, subject_ids = load_data(
        cfg["data"]["train_h5"],
        cfg["data"]["train_ids_csv"],
        cfg["data"]["train_labels_csv"],
    )
    print(f"  Signals: {signals.shape}, Labels: {labels.shape}")
    print(f"  Subjects: {np.unique(subject_ids).tolist()}")

    # ── Train/val split ───────────────────────────────────────────────────────
    if args.all_subjects:
        print(f"  Mode ALL SUBJECTS — {len(signals)} samples, {np.unique(subject_ids).shape[0]} subjects")
        print(f"  Epochs fixes : {cfg['training']['n_epochs']}")
        X_tr, y_tr = signals, labels
        X_val, y_val = None, None
    else:
        X_tr, y_tr, sub_tr, X_val, y_val, sub_val = subject_train_val_split(
            signals, labels, subject_ids,
            val_ratio=cfg["data"]["val_ratio"],
            seed=cfg["data"]["seed"],
        )
        print(f"  Train: {len(X_tr)} samples ({np.unique(sub_tr).shape[0]} subjects)")
        print(f"  Val:   {len(X_val)} samples ({np.unique(sub_val).shape[0]} subjects)")

    # ── Normalize ─────────────────────────────────────────────────────────────
    if cfg["data"].get("instance_norm", False):
        print("Using instance normalization (per window)...")
        X_tr = clip_outliers(instance_normalize(X_tr))
        if X_val is not None:
            X_val = clip_outliers(instance_normalize(X_val))
    else:
        print("Computing normalization stats on training set...")
        mean, std = compute_stats(X_tr)
        save_stats(mean, std, cfg["data"]["stats_path"])
        X_tr = clip_outliers(normalize(X_tr, mean, std))
        if X_val is not None:
            X_val = clip_outliers(normalize(X_val, mean, std))

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_transform = get_train_transform() if cfg["training"]["augment"] else None
    train_ds = SleepApneaDataset(X_tr, y_tr, transform=train_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = None
    if X_val is not None:
        val_ds = SleepApneaDataset(X_val, y_val)
        val_loader = DataLoader(
            val_ds,
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

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg['model']['name']} | Parameters: {n_params:,}")

    # ── Optimizer & loss ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    loss_fn = build_loss(
        cfg["loss"]["name"],
        **{k: v for k, v in cfg["loss"].items() if k != "name"},
    )
    scheduler = build_scheduler(optimizer, cfg)

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=cfg["output"]["checkpoint_dir"],
        scheduler=scheduler,
        grad_clip=cfg["training"]["grad_clip"],
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=cfg["training"]["n_epochs"],
        patience=cfg["training"]["patience"],
        model_name=cfg["output"]["model_name"],
    )


if __name__ == "__main__":
    main()
