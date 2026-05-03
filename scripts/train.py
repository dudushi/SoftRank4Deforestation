"""Train one model with one loss.

Usage
-----
    python -m scripts.train --loss softrank_plus --data-root data/

`--loss` ∈ {focal, wmse, ltk, softrank, softrank_plus}.
The trained checkpoint goes to `--checkpoint-dir/<loss>.pt` and the
inference output to `--results-dir/<loss>.pt` plus a single shared
`targets.pt`.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import torch

from softrank import config as C
from softrank import losses as L
from softrank import utils as U
from softrank.config import default_config
from softrank.data import build_dataloaders
from softrank.inference import run_inference
from softrank.models import build_model
from softrank.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loss", required=True, choices=L.LOSS_NAMES)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--results-dir",   type=Path, default=Path("results"))
    p.add_argument("--seed",          type=int,  default=C.SEED)
    p.add_argument("--epochs",        type=int,  default=None,
                   help="override config.EPOCHS for quick tests")
    p.add_argument("--device",        default=None,
                   help="override config.DEVICE (cuda / cpu)")
    p.add_argument("--skip-inference", action="store_true",
                   help="train only, do not run test inference afterwards")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = default_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device

    cfg.seed = args.seed
    U.set_seed(cfg.seed)

    # ---- data ----
    train_loader, val_loader, test_loader, _ = build_dataloaders(
        args.data_root,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    # ---- model ----
    model = build_model(cfg).to(cfg.device)

    # ---- train ----
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint_dir / f"{args.loss}.pt"
    trainer = Trainer(model, cfg, args.loss, train_loader, val_loader, ckpt)
    best = trainer.fit()
    print(f"\nBest val NDCG@{cfg.K} for {args.loss}: {best:.6f}")

    if args.skip_inference:
        return 0

    # ---- inference ----
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n→ Loading best checkpoint and running test inference …")
    U.load_checkpoint(str(ckpt), model, map_location=cfg.device)

    probs, targets, _ = run_inference(model, test_loader, device=cfg.device)
    torch.save(probs,   args.results_dir / f"{args.loss}.pt")
    if not (args.results_dir / "targets.pt").is_file():
        torch.save(targets, args.results_dir / "targets.pt")
    print(f"✓ saved {args.results_dir / (args.loss + '.pt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
