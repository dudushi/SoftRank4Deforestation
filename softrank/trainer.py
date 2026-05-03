"""Unified trainer.

One class, `Trainer`, replaces the old split between `train.py` and
`train_softrank.py`. It dispatches the per-loss forward signatures (some
losses take 3 arguments, others 4) and uses the right ReduceLROnPlateau
patience per family. The validation metric is **always** NDCG@K so that
models trained with different losses are directly comparable.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as C
from . import losses as L
from . import metrics as M
from . import utils as U


# ---------------------------------------------------------------------------
# Per-loss forward signatures
# ---------------------------------------------------------------------------
# focal / softrank / softrank_plus : (pred, target, biome)
# wmse                              : (pred, target, biome, deforest_mask)
# ltk                               : (pred, target, biome, TK_value)
# All five operate on the *inner* (cropped) tensor.
def _loss_forward(loss_fn: nn.Module,
                  predictions: torch.Tensor,
                  y: torch.Tensor,
                  biome: torch.Tensor,
                  TK_value: torch.Tensor,
                  loss_name: str) -> torch.Tensor:
    p = U.crop_inner(predictions)
    t = U.crop_inner(y)
    b = U.crop_inner(biome)
    if loss_name == "wmse":
        m = (t > 0).float()
        return loss_fn(p, t, b, m)
    if loss_name == "ltk":
        return loss_fn(p, t, b, TK_value)
    return loss_fn(p, t, b)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Train a `ResUNet` for one of the five paper losses.

    Parameters
    ----------
    model         : ResUNet (already on `cfg.device`).
    cfg           : `softrank.config.Config`.
    loss_name     : one of softrank.losses.LOSS_NAMES.
    train_loader  : DataLoader yielding (x, y, biome, coords, TK_value).
    val_loader    : DataLoader of the same shape.
    checkpoint_path : where the best (highest val-NDCG@K) state is saved.
    """

    def __init__(self,
                 model: nn.Module,
                 cfg,
                 loss_name: str,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 checkpoint_path: str | Path):
        if loss_name not in L.LOSS_NAMES:
            raise ValueError(f"Unknown loss '{loss_name}'.")
        self.model = model
        self.cfg = cfg
        self.loss_name = loss_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_path = Path(checkpoint_path)

        self.device = cfg.device
        self.loss_fn = L.build_loss(loss_name, cfg).to(cfg.device)

        self.optimizer = optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        # Different patience for ranking vs regression families — paper-faithful.
        sched_patience = (cfg.sched_patience_rank
                          if loss_name in ("softrank", "softrank_plus")
                          else cfg.sched_patience_regression)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max",
            factor=cfg.sched_factor,
            threshold=cfg.sched_threshold,
            patience=sched_patience,
            min_lr=cfg.sched_min_lr,
        )

    # ----------------------------------------------------------------- train
    def fit(self) -> float:
        """Run training; return the best val NDCG@K."""
        best_val = float("-inf")
        patience_counter = 0
        for epoch in range(1, self.cfg.epochs + 1):
            print("🌳" * 20)
            print(f"[EPOCH {epoch:03d}/{self.cfg.epochs:03d}]   loss = {self.loss_name}")
            print("🌳" * 20)

            self._train_one_epoch()
            val_ndcg = self.validate()

            print(f"  Val NDCG@{self.cfg.K}: {val_ndcg:.6f}")
            self.scheduler.step(val_ndcg)
            cur_lr = self.optimizer.param_groups[0]["lr"]
            print(f"  Learning rate: {cur_lr:.2e}")

            if val_ndcg > best_val:
                best_val = val_ndcg
                patience_counter = 0
                self._save_checkpoint(epoch, val_ndcg)
                print(f"  ✓ new best — saved to {self.checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}.")
                    break
        return best_val

    # --------------------------------------------------------- internal loops
    def _train_one_epoch(self) -> None:
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        for x, y, biome, _, TK_value in tqdm(self.train_loader, desc="Training"):
            x, y, biome = (x.to(self.device, dtype=torch.float32),
                           y.to(self.device, dtype=torch.float32),
                           biome.to(self.device))
            TK_value = TK_value.to(self.device, dtype=torch.float32)

            preds = self.model(x)
            loss = _loss_forward(self.loss_fn, preds, y, biome, TK_value,
                                 self.loss_name)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1
        avg = total_loss / max(total_batches, 1)
        print(f"  Train loss : {avg:.6f}")

    @torch.no_grad()
    def validate(self) -> float:
        """Validation metric is **NDCG@K** for every loss (paper §VI-B)."""
        self.model.eval()
        total = 0.0
        for x, y, biome, _, _ in tqdm(self.val_loader, desc="Validating"):
            x = x.to(self.device, dtype=torch.float32)
            y = y.to(self.device, dtype=torch.float32)
            preds = self.model(x)
            inner_p = U.crop_inner(preds).reshape(preds.shape[0], -1)
            inner_y = U.crop_inner(y).reshape(y.shape[0], -1)
            # We need a per-batch NDCG, not a flattened one over T:
            for bi in range(inner_p.shape[0]):
                total += M.ndcg_at_k(inner_p[bi:bi + 1], inner_y[bi:bi + 1],
                                     k=self.cfg.K)
        return total / len(self.val_loader.dataset)

    # --------------------------------------------------------- checkpointing
    def _save_checkpoint(self, epoch: int, val_ndcg: float) -> None:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "loss_name": self.loss_name,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "val_ndcg": val_ndcg,
            },
            self.checkpoint_path,
        )
