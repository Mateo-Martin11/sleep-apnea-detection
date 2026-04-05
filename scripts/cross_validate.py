"""5-fold cross-validation par sujet pour calibrer threshold et min_duration.

Workflow :
1. GroupKFold(5) sur les 22 sujets
2. Pour chaque fold : entraîne sur ~18 sujets, prédit sur ~4
3. Agrège toutes les prédictions OOF (out-of-fold)
4. Sweep threshold x min_duration sur l'ensemble OOF
5. Affiche le meilleur couple (threshold, min_duration)

Usage :
    python scripts/cross_validate.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import load_data, SleepApneaDataset
from src.data.preprocessing import clip_outliers, instance_normalize
from src.data.augmentation import get_train_transform
from src.models.factory import build_model
from src.training.losses import build_loss
from src.training.trainer import Trainer
from src.training.metrics import compute_f1_events, threshold_predictions, apply_postprocessing


def train_fold(fold_idx, train_idx, val_idx, signals, labels, cfg, device):
    X_tr, y_tr = signals[train_idx], labels[train_idx]
    X_val, y_val = signals[val_idx], labels[val_idx]

    X_tr  = clip_outliers(instance_normalize(X_tr))
    X_val = clip_outliers(instance_normalize(X_val))

    train_transform = get_train_transform() if cfg["training"]["augment"] else None
    train_ds = SleepApneaDataset(X_tr, y_tr, transform=train_transform)
    val_ds   = SleepApneaDataset(X_val)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"] * 2,
                              shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(
        name=cfg["model"]["name"],
        n_channels=cfg["model"]["n_channels"],
        base_filters=cfg["model"]["base_filters"],
        dropout=cfg["model"].get("dropout", 0.2),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    loss_fn = build_loss(
        cfg["loss"]["name"],
        **{k: v for k, v in cfg["loss"].items() if k != "name"},
    )

    # Entraîner pour n_epochs fixes (pas d'early stopping en CV)
    n_epochs = cfg["training"]["n_epochs"]
    trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn,
                      device=device, checkpoint_dir="checkpoints",
                      grad_clip=cfg["training"]["grad_clip"])

    print(f"\n--- Fold {fold_idx+1}/5 | train={len(X_tr)} val={len(X_val)} ---")
    for epoch in range(1, n_epochs + 1):
        loss = trainer.train_epoch(train_loader)
        print(f"  Epoch {epoch:02d}/{n_epochs} | loss={loss:.4f}")

    # Prédictions OOF
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"  predict fold {fold_idx+1}", leave=False):
            x = batch if not isinstance(batch, (list, tuple)) else batch[0]
            all_probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())

    return np.concatenate(all_probs), y_val


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    signals, labels, _, subject_ids = load_data(
        cfg["data"]["train_h5"],
        cfg["data"]["train_ids_csv"],
        cfg["data"]["train_labels_csv"],
    )
    print(f"Signals: {signals.shape} | Labels: {labels.shape}")
    print(f"Subjects: {np.unique(subject_ids).tolist()}")

    # ── 5-fold CV par sujet ───────────────────────────────────────────────────
    gkf = GroupKFold(n_splits=5)
    idx = np.arange(len(signals))

    oof_probs  = np.zeros_like(labels, dtype=np.float32)
    oof_labels = np.zeros_like(labels, dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(idx, groups=subject_ids)):
        probs, y_val = train_fold(fold_idx, train_idx, val_idx, signals, labels, cfg, device)
        oof_probs[val_idx]  = probs
        oof_labels[val_idx] = y_val

    # ── Sweep threshold x min_duration sur OOF complet ───────────────────────
    print("\n\nSweep threshold x min_duration sur prédictions OOF (4400 fenêtres) :")
    print(f"{'thr':>6} {'min_dur':>8} {'F1':>8} {'P':>8} {'R':>8}")

    best_f1, best_thr, best_min_dur = 0.0, 0.5, 9
    for thr in np.arange(0.40, 0.85, 0.05):
        for min_dur in range(5, 16):
            y_pred = threshold_predictions(oof_probs, thr)
            y_pp   = apply_postprocessing(y_pred, min_duration=min_dur)
            f1, p, r = compute_f1_events(oof_labels, y_pp)
            if f1 > best_f1:
                best_f1, best_thr, best_min_dur = f1, round(float(thr), 2), min_dur
                print(f"{thr:>6.2f} {min_dur:>8} {f1:>8.4f} {p:>8.4f} {r:>8.4f}  <- best")

    print(f"\nMeilleur couple : threshold={best_thr}, min_duration={best_min_dur}")
    print(f"F1 OOF          : {best_f1:.4f}")
    print(f"\nUtilise ces valeurs pour la soumission finale :")
    print(f"  python scripts/train.py --all_subjects")
    print(f"  python scripts/predict.py --threshold {best_thr} --min_duration {best_min_dur}")

    # Sauvegarder les résultats
    np.savez("checkpoints/oof_results.npz",
             probs=oof_probs, labels=oof_labels,
             best_thr=best_thr, best_min_dur=best_min_dur)
    print(f"\nRésultats OOF sauvegardés dans checkpoints/oof_results.npz")


if __name__ == "__main__":
    main()
