"""Recompute Priority@K / NDCG@K from already-saved inference tensors.

Useful when you keep `results/<loss>.pt` from a previous run and only
want to refresh the table.

Usage
-----
    python -m scripts.evaluate --results-dir results/
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

from softrank import config as C
from softrank import losses as L
from softrank import metrics as M


KS = (5, 10, 30, 50, 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--targets", type=Path, default=None,
                   help="path to targets.pt (default: results-dir/targets.pt)")
    return p.parse_args()


def _crop_pad(stack: torch.Tensor) -> torch.Tensor:
    return stack[:, C.SIZE:-C.SIZE, C.SIZE:-C.SIZE]


def main() -> int:
    args = parse_args()
    targets_path = args.targets or (args.results_dir / "targets.pt")
    if not targets_path.is_file():
        print(f"error: missing {targets_path}", file=sys.stderr)
        return 1
    targets = _crop_pad(torch.load(targets_path, map_location="cpu"))

    table: dict = {}
    for loss_name in L.LOSS_NAMES:
        path = args.results_dir / f"{loss_name}.pt"
        if not path.is_file():
            print(f"  · {loss_name:<14} — skipped (no {path.name})")
            continue
        probs = _crop_pad(torch.load(path, map_location="cpu"))
        table[loss_name] = M.evaluate_all_K(probs, targets, ks=KS)
        print(f"  · {loss_name:<14} ✓")

    print()
    for metric, label in (("priority_at_k", "Priority@K"),
                          ("ndcg_at_k",     "NDCG@K")):
        print(f"  ── {label} ──")
        header = "Loss".ljust(16) + "".join(f"K={k:<8}" for k in KS)
        print(header)
        print("-" * len(header))
        for loss_name, scores in table.items():
            row = loss_name.ljust(16)
            for k in KS:
                row += f"{scores[metric][k]:.3f}     "
            print(row)
        print()

    out = args.results_dir / "evaluate.json"
    with open(out, "w") as f:
        json.dump(table, f, indent=2)
    print(f"→ Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
