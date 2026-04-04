"""Training loop with early stopping and checkpoint management."""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_f1_events, threshold_predictions, find_best_threshold


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        device: torch.device,
        checkpoint_dir: str = "checkpoints",
        scheduler=None,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_f1 = 0.0
        self.best_threshold = 0.5
        self.history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_precision": [], "val_recall": []}

    # ── Training ──────────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for x, y in tqdm(loader, desc="  train", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item() * len(x)

        return total_loss / len(loader.dataset)

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_probs, all_targets = [], []

        for x, y in tqdm(loader, desc="  val  ", leave=False):
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            total_loss += loss.item() * len(x)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y.cpu().numpy())

        probs = np.concatenate(all_probs)        # (N, 90)
        targets = np.concatenate(all_targets)    # (N, 90)

        # Find best threshold on validation set
        threshold, f1 = find_best_threshold(targets, probs)
        y_pred = threshold_predictions(probs, threshold)
        _, precision, recall = compute_f1_events(targets, y_pred)

        return {
            "val_loss": total_loss / len(loader.dataset),
            "val_f1": f1,
            "val_precision": precision,
            "val_recall": recall,
            "threshold": threshold,
        }

    # ── Full training loop ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        n_epochs: int,
        patience: int = 10,
        model_name: str = "model",
    ):
        """Train the model.

        If val_loader is None, trains for exactly n_epochs on all data
        and saves the final checkpoint (no early stopping).
        """
        no_improve = 0
        all_subjects_mode = val_loader is None

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            elapsed = time.time() - t0

            if all_subjects_mode:
                if self.scheduler is not None and not isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()
                print(f"Epoch {epoch:03d}/{n_epochs} | loss={train_loss:.4f} | {elapsed:.1f}s")
                self.history["train_loss"].append(train_loss)
                continue

            # ── With validation ───────────────────────────────────────────────
            metrics = self.validate(val_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics["val_f1"])
                else:
                    self.scheduler.step()

            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"loss={train_loss:.4f} | "
                f"val_loss={metrics['val_loss']:.4f} | "
                f"F1={metrics['val_f1']:.4f} "
                f"(P={metrics['val_precision']:.3f} R={metrics['val_recall']:.3f}) | "
                f"thr={metrics['threshold']:.2f} | "
                f"{elapsed:.1f}s"
            )

            self.history["train_loss"].append(train_loss)
            for k in ["val_loss", "val_f1", "val_precision", "val_recall"]:
                self.history[k].append(metrics[k])

            if metrics["val_f1"] > self.best_f1:
                self.best_f1 = metrics["val_f1"]
                self.best_threshold = metrics["threshold"]
                no_improve = 0
                ckpt_path = self.checkpoint_dir / f"{model_name}_best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": self.model.state_dict(),
                        "optimizer_state": self.optimizer.state_dict(),
                        "best_f1": self.best_f1,
                        "best_threshold": self.best_threshold,
                    },
                    ckpt_path,
                )
                print(f"  New best F1={self.best_f1:.4f} — saved to {ckpt_path}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping after {epoch} epochs (no improvement for {patience} epochs).")
                    break

        # ── Save final checkpoint (all-subjects mode) ─────────────────────────
        if all_subjects_mode:
            ckpt_path = self.checkpoint_dir / f"{model_name}_final.pt"
            torch.save(
                {
                    "epoch": n_epochs,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "best_f1": None,
                    "best_threshold": 0.60,  # use threshold from val run
                },
                ckpt_path,
            )
            print(f"\nTraining complete. Checkpoint saved to {ckpt_path}")
        else:
            print(f"\nTraining complete. Best val F1: {self.best_f1:.4f} at threshold={self.best_threshold:.2f}")

        return self.history

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, loader: DataLoader, threshold: float = None) -> np.ndarray:
        """Return binary predictions for all samples in loader."""
        if threshold is None:
            threshold = self.best_threshold

        self.model.eval()
        all_probs = []

        for batch in tqdm(loader, desc="predict"):
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(self.device)
            logits = self.model(x)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

        probs = np.concatenate(all_probs)
        return threshold_predictions(probs, threshold), probs
