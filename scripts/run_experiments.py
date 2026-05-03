"""Run the full Table-I experiment in one shot.

Trains all five losses sequentially (Focal → WMSE → LTK → SoftRank →
SoftRank+), runs inference for each, and prints / saves the final
Priority@K & NDCG@K table.

Usage
-----
    python -m scripts.run_experiments --data-root data/
    python -m scripts.run_experiments --data-root data/ --losses softrank softrank_plus --seeds 0 1 2

A single seed reproduces a single column of Table I; pass `--seeds 0 1 2 3 4 5`
to mirror the paper's six-run statistics. Per-seed checkpoints are written
to `checkpoints/<loss>_seed<n>.pt`.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from softrank import config as C
from softrank import losses as L
from softrank import metrics as M
from softrank import utils as U
from softrank.config import default_config
from softrank.data import build_dataloaders
from softrank.inference import run_inference
from softrank.models import build_model
from softrank.trainer import Trainer


KS = (5, 10, 30, 50, 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--results-dir",   type=Path, default=Path("results"))
    p.add_argument("--losses", nargs="+", default=list(L.LOSS_NAMES),
                   choices=L.LOSS_NAMES)
    p.add_argument("--seeds",  nargs="+", type=int, default=[C.SEED])
    p.add_argument("--epochs", type=int, default=None,
                   help="override config.EPOCHS for quick smoke tests")
    p.add_argument("--device", default=None)
    return p.parse_args()


def _crop_pad(stack: torch.Tensor) -> torch.Tensor:
    """Strip the SIZE-pixel border that was added during preprocessing."""
    return stack[:, C.SIZE:-C.SIZE, C.SIZE:-C.SIZE]


def train_one(loss_name: str, seed: int, args, cfg, loaders) -> Path:
    """Train one loss/seed combo and return the inference-result path."""
    train_loader, val_loader, test_loader, _ = loaders

    cfg.seed = seed
    U.set_seed(seed)

    model = build_model(cfg).to(cfg.device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.checkpoint_dir / f"{loss_name}_seed{seed}.pt"
    trainer = Trainer(model, cfg, loss_name, train_loader, val_loader, ckpt)
    trainer.fit()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    U.load_checkpoint(str(ckpt), model, map_location=cfg.device)
    probs, targets, _ = run_inference(model, test_loader, device=cfg.device)
    out = args.results_dir / f"{loss_name}_seed{seed}.pt"
    torch.save(probs, out)
    if not (args.results_dir / "targets.pt").is_file():
        torch.save(targets, args.results_dir / "targets.pt")
    return out


def evaluate(probs_path: Path, targets_path: Path) -> dict:
    """Run Priority@K / NDCG@K at every K reported in the paper."""
    probs   = _crop_pad(torch.load(probs_path,  map_location="cpu"))
    targets = _crop_pad(torch.load(targets_path, map_location="cpu"))
    return M.evaluate_all_K(probs, targets, ks=KS)


def aggregate(per_seed_runs: List[dict]) -> dict:
    """Mean / std across seeds for every (metric, K)."""
    agg: dict = {"priority_at_k": {}, "ndcg_at_k": {}}
    for metric in ("priority_at_k", "ndcg_at_k"):
        for k in KS:
            vals = [run[metric][k] for run in per_seed_runs]
            agg[metric][k] = {"mean": float(np.mean(vals)),
                              "std":  float(np.std(vals))}
    return agg


def print_table(per_loss: dict) -> None:
    """Pretty-print the paper's Table I."""
    for metric, label in (("priority_at_k", "Priority@K"),
                          ("ndcg_at_k",     "NDCG@K")):
        print()
        print(f"  ── {label} ──")
        header = "Loss".ljust(16) + "".join(f"K={k:<8}" for k in KS)
        print(header)
        print("-" * len(header))
        for loss_name, agg in per_loss.items():
            row = loss_name.ljust(16)
            for k in KS:
                m, s = agg[metric][k]["mean"], agg[metric][k]["std"]
                row += f"{m:.3f}±{s:.3f}  "
            print(row)


def main() -> int:
    args = parse_args()
    cfg = default_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.device is not None:
        cfg.device = args.device

    print(f"→ Building dataloaders once (re-used across {len(args.losses)} loss(es) "
          f"× {len(args.seeds)} seed(s)) …")
    loaders = build_dataloaders(
        args.data_root, batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)

    per_loss: dict = {}
    targets_path = args.results_dir / "targets.pt"

    for loss_name in args.losses:
        per_seed = []
        for seed in args.seeds:
            print("\n" + "=" * 78)
            print(f"  TRAIN  loss={loss_name}  seed={seed}")
            print("=" * 78)
            probs_path = train_one(loss_name, seed, args, cfg, loaders)
            per_seed.append(evaluate(probs_path, targets_path))
        per_loss[loss_name] = aggregate(per_seed)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.results_dir / "table_I.json"
    with open(out_json, "w") as f:
        json.dump(per_loss, f, indent=2)

    print("\n" + "=" * 78)
    print("  RESULTS  (mean ± std over seeds)")
    print("=" * 78)
    print_table(per_loss)
    print(f"\n→ Full numbers written to {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
