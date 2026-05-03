"""Sliding-biweek inference over the held-out test split.

Produces:
    probs   (T, H, W)  — the network's predicted score map per biweek
    targets (T, H, W)  — the corresponding ground-truth deforestation map
    mask    (H, W)     — the biome mask used during inference

The output spatial size matches the *padded* DETER-B grid; pass through
`softrank.utils.crop_inner` (or strip `SIZE` cells on every side) to
return to the natural grid expected by `softrank.metrics`.
"""

from __future__ import annotations
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as C
from . import utils as U


@torch.no_grad()
def run_inference(model: torch.nn.Module,
                  test_loader: DataLoader,
                  device: str = C.DEVICE
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Iterate every test biweek and stack the predictions."""
    model = model.to(device).eval()

    probs   = torch.empty(C.TEST_BIWEEKS,
                          C.NORMAL_DETER_SIZE["y"] + 2 * C.SIZE,
                          C.NORMAL_DETER_SIZE["x"] + 2 * C.SIZE)
    targets = torch.empty_like(probs)

    last_mask: torch.Tensor | None = None

    for i, (x, y, mask) in enumerate(tqdm(test_loader, desc="Inference")):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32).squeeze()
        mask = mask.to(device)

        x = U.pad_inference(x)
        logits = model(x)
        logits = U.unpad_inference(logits)

        probs[i]   = logits.squeeze().cpu()
        targets[i] = y.cpu()
        last_mask  = mask.cpu()

    assert last_mask is not None, "test_loader produced no batches"
    return probs, targets, last_mask
