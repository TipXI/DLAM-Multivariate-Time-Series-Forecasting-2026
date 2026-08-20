"""Training and evaluation harness for PyTorch forecasting models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation import compute_all_metrics


class Trainer:
    """Trains forecasting models, monitors validation WAPE, and manages checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        loss_type: str = "l1",  # "l1" or "huber" or "mse"
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: Path | str = "checkpoints",
        model_name: str = "model",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        if loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.HuberLoss(delta=1.0)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

        self.best_val_wape = float("inf")
        self.best_checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.pt"

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            past_target = batch["past_target"].to(self.device)
            future_target = batch["future_target"].to(self.device)

            kwargs = {
                "past_target": past_target,
            }
            if "past_covariates" in batch:
                kwargs["past_covariates"] = batch["past_covariates"].to(self.device)
            if "future_covariates" in batch:
                kwargs["future_covariates"] = batch["future_covariates"].to(self.device)
            if "series_idx" in batch:
                kwargs["series_idx"] = batch["series_idx"].to(self.device)
            if "past_seasonal" in batch:
                kwargs["past_seasonal"] = batch["past_seasonal"].to(self.device)
            if "future_seasonal" in batch:
                kwargs["future_seasonal"] = batch["future_seasonal"].to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(**kwargs)
            loss = self.criterion(pred, future_target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_trues = []

        for batch in val_loader:
            past_target = batch["past_target"].to(self.device)
            future_target = batch["future_target"].to(self.device)

            kwargs = {
                "past_target": past_target,
            }
            if "past_covariates" in batch:
                kwargs["past_covariates"] = batch["past_covariates"].to(self.device)
            if "future_covariates" in batch:
                kwargs["future_covariates"] = batch["future_covariates"].to(self.device)
            if "series_idx" in batch:
                kwargs["series_idx"] = batch["series_idx"].to(self.device)
            if "past_seasonal" in batch:
                kwargs["past_seasonal"] = batch["past_seasonal"].to(self.device)
            if "future_seasonal" in batch:
                kwargs["future_seasonal"] = batch["future_seasonal"].to(self.device)

            pred = self.model(**kwargs)

            all_preds.append(pred.cpu().numpy().flatten())
            all_trues.append(future_target.cpu().numpy().flatten())

        y_p = np.concatenate(all_preds)
        y_t = np.concatenate(all_trues)

        metrics = compute_all_metrics(y_t, y_p)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        early_stopping_patience: int = 8,
    ) -> Dict[str, float]:
        print(f"\nStarting training on {self.device.upper()} for {epochs} epochs...")
        print(f"Model: {self.model_name} | Best checkpoint destination: {self.best_checkpoint_path}")
        patience_counter = 0
        best_metrics = {}

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_wape = val_metrics["WAPE"]
            elapsed = time.time() - t0

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.1f}s] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val WAPE: {val_wape:.4f} | "
                f"Val MAE: {val_metrics['MAE']:.3f} | "
                f"Val RMSE: {val_metrics['RMSE']:.3f} | "
                f"LR: {lr:.6f}",
                flush=True,
            )

            if val_wape < self.best_val_wape:
                self.best_val_wape = val_wape
                best_metrics = val_metrics
                patience_counter = 0

                # Save checkpoint with model weights and configuration
                checkpoint_dict = {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "val_wape": val_wape,
                    "metrics": val_metrics,
                }
                torch.save(checkpoint_dict, self.best_checkpoint_path)
                print(f"  --> Saved new best checkpoint (Val WAPE: {val_wape:.4f})", flush=True)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs (no improvement for {patience_counter} epochs).", flush=True)
                    break

        print(f"\nTraining Complete! Best Validation WAPE: {self.best_val_wape:.4f}", flush=True)
        return best_metrics
