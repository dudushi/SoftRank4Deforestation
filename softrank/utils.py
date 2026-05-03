"""Reproducibility, sparse-cube I/O, and small numerical helpers.

Nothing in this module is paper-specific; it just wraps the bookkeeping
needed by `data.py` and `trainer.py`.
"""

from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from . import config as C


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int = C.SEED) -> None:
    """Pin every RNG we touch and disable cuDNN nondeterminism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Sparse cube I/O
# ---------------------------------------------------------------------------
def load_sparse_npz(path: str, start: int = 0) -> List[csr_matrix]:
    """Load a sequence of sparse biweekly matrices from an .npz archive.

    Each entry in the archive is either a 0-d object array containing a
    sparse matrix, or already an ndarray/csr.
    `start > 0` truncates the head of the cube so it aligns with the
    seasonality cube (which has fewer biweeks).
    """
    data = np.load(path, allow_pickle=True)
    out: List[csr_matrix] = []
    for key in sorted(data.files):
        arr = data[key]
        out.append(arr.item() if arr.shape == () else arr)
    return out[-start:] if start else out


def pad_sparse_cube(matrices: Sequence[csr_matrix], pad_size: int) -> List[csr_matrix]:
    """Zero-pad each frame of a sparse cube on all four sides by `pad_size`."""
    padded: List[csr_matrix] = []
    for m in matrices:
        dense = m.toarray()
        dense = np.pad(dense, ((pad_size, pad_size), (pad_size, pad_size)),
                       mode="constant")
        padded.append(csr_matrix(dense))
    return padded


# ---------------------------------------------------------------------------
# Top-K threshold τ_K^{(t)} (paper Eq. 2)
# ---------------------------------------------------------------------------
def compute_global_TK(deter_area: Sequence[csr_matrix], K: int = C.K
                      ) -> Dict[int, float]:
    """For each biweek, compute the K-th largest deforestation intensity.

    Returns a dict {biweek_index → τ_K^{(t)}} consumed by `LTKLoss` and the
    Deterset dataloader.
    """
    out: Dict[int, float] = {}
    for t in range(len(deter_area)):
        arr = deter_area[t].toarray() if hasattr(deter_area[t], "toarray") \
              else np.asarray(deter_area[t])
        flat = arr.ravel()
        if np.all(flat == 0):
            out[t] = 0.0
            continue
        kth = max(0, flat.size - K)
        out[t] = float(np.partition(flat, kth)[kth])
    return out


# ---------------------------------------------------------------------------
# Inner crop (boundary cells excluded from loss) — paper §VI-A
# ---------------------------------------------------------------------------
def crop_inner(tensor: torch.Tensor, border: int = C.INNER_BORDER) -> torch.Tensor:
    """Strip `border` pixels from every spatial side of a (B,C,H,W) or (B,H,W) tensor."""
    sl = slice(border, -border)
    if tensor.dim() == 4:
        return tensor[:, :, sl, sl]
    if tensor.dim() == 3:
        return tensor[:, sl, sl]
    raise ValueError(f"crop_inner expects a 3D or 4D tensor; got dim={tensor.dim()}.")


# ---------------------------------------------------------------------------
# Inference padding (kept identical to original code)
# ---------------------------------------------------------------------------
_C_PAD = 3   # right-padding so width is divisible by 2 enough times for the U-Net


def pad_inference(x: torch.Tensor) -> torch.Tensor:
    """Append `_C_PAD` zero columns on the right so H/W are friendly to the U-Net."""
    cols = torch.zeros((x.shape[0], x.shape[1], x.shape[2], _C_PAD),
                       device=x.device, dtype=x.dtype)
    return torch.cat([x, cols], dim=3)


def unpad_inference(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, :, :-_C_PAD]


def match_and_cat(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    """Center-pad `upsampled` to match `skip` and concatenate along channels."""
    diffY = skip.size(2) - upsampled.size(2)
    diffX = skip.size(3) - upsampled.size(3)
    if diffX or diffY:
        upsampled = F.pad(upsampled,
                          [diffX // 2, diffX - diffX // 2,
                           diffY // 2, diffY - diffY // 2])
    return torch.cat([upsampled, skip], dim=1)


# ---------------------------------------------------------------------------
# Biweek calendar (kept for downstream notebooks / qualitative plots)
# ---------------------------------------------------------------------------
def build_biweekly_index(start: str = C.STARTING_DATE,
                         end: str = C.ENDING_DATE,
                         M: int = C.M) -> List[Tuple[str, str, int]]:
    """List of (current_biweek, predicted_biweek_M-ahead, season_idx)."""
    biweek_to_idx = {f"{d:02d}/{m:02d}": (m - 1) * 2 + (1 if d == 1 else 2)
                     for m in range(1, 13) for d in (1, 16)}
    init = datetime.strptime(start, "%d/%m/%Y")
    final = datetime.strptime(end,   "%d/%m/%Y")

    out: List[Tuple[str, str, int]] = []
    cur = init
    while cur <= final:
        if cur.day not in (1, 16):
            cur += timedelta(days=1)
            continue
        cur_label = cur.strftime("%d/%m/%Y")

        # advance M biweeks
        future = cur
        for _ in range(M):
            if future.day == 1:
                future = future.replace(day=16)
            else:
                future = future.replace(
                    year=future.year + (1 if future.month == 12 else 0),
                    month=1 if future.month == 12 else future.month + 1,
                    day=1,
                )
        out.append((cur_label, future.strftime("%d/%m/%Y"),
                    biweek_to_idx[future.strftime("%d/%m")]))

        # advance to next biweek
        if cur.day == 1:
            cur = cur.replace(day=16)
        else:
            cur = cur.replace(
                year=cur.year + (1 if cur.month == 12 else 0),
                month=1 if cur.month == 12 else cur.month + 1,
                day=1,
            )
    return out


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
def save_checkpoint(state: dict, filename: str) -> None:
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(path: str, model: torch.nn.Module,
                    map_location: str = C.DEVICE) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    return model
