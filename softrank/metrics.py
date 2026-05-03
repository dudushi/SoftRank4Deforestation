"""Evaluation metrics defined in §II of the paper.

Three metrics are exposed:

    priority_at_k        — set-overlap of predicted Top-K with GT Top-K (Eq. 3).
    ndcg_at_k            — log-discounted ranking quality (Eq. 4–5).
    pairwise_distance_cdf — spatial-dispersion CDF of correctly detected pairs.

All three operate on either:
    – a single biweek    (H, W)
    – a stack of biweeks (T, H, W)

When given (T, H, W), priority_at_k / ndcg_at_k average over T.
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k flat indices of (..., H, W) with NaNs/+Inf masked out."""
    flat = x.reshape(*x.shape[:-2], -1).clone()
    flat[~torch.isfinite(flat)] = -float("inf")
    return torch.topk(flat, k=k, dim=-1).indices


# ---------------------------------------------------------------------------
# Priority@K  — paper Eq. 3
# ---------------------------------------------------------------------------
def priority_at_k(prediction: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """|Ŝ_K ∩ S*_K| / K  averaged over all biweeks in the stack."""
    if prediction.dim() == 2:
        prediction, target = prediction.unsqueeze(0), target.unsqueeze(0)

    pred_idx = _topk_indices(prediction, k)
    true_idx = _topk_indices(target,    k)

    overlaps = []
    for p, t in zip(pred_idx, true_idx):
        inter = torch.isin(p, t).sum().item()
        overlaps.append(inter / k)
    return float(np.mean(overlaps))


# ---------------------------------------------------------------------------
# NDCG@K  — paper Eq. 4–5
# ---------------------------------------------------------------------------
def ndcg_at_k(prediction: torch.Tensor, target: torch.Tensor, k: int) -> float:
    """Mean NDCG@K over the time dimension. Handles shape (H,W) or (T,H,W)."""
    if prediction.dim() == 2:
        prediction, target = prediction.unsqueeze(0), target.unsqueeze(0)

    B = prediction.shape[0]
    pred_flat   = prediction.reshape(B, -1).float()
    target_flat = target.reshape(B, -1).float()

    _, top_k_idx = torch.topk(pred_flat, k, dim=1)
    relevant = torch.gather(target_flat, 1, top_k_idx)            # (B, K)

    ranks = torch.arange(2, k + 2, device=prediction.device).float()
    discounts = torch.log2(ranks)                                 # 1-indexed log2
    dcg = (relevant / discounts).sum(dim=1)

    sorted_relevance, _ = torch.sort(target_flat, dim=1, descending=True)
    idcg = (sorted_relevance[:, :k] / discounts).sum(dim=1)

    return float((dcg / (idcg + 1e-8)).mean().item())


# ---------------------------------------------------------------------------
# Pairwise Distance CDF  — paper §II-D
# ---------------------------------------------------------------------------
def pairwise_distance_cdf(prediction: torch.Tensor,
                          target: torch.Tensor,
                          k: int,
                          distances: np.ndarray) -> np.ndarray:
    """For each search radius d, fraction of correctly-detected pairs ≤ d.

    Aggregates across all biweeks in the stack. Returns an array F(d) with
    the same length as `distances`.
    """
    if prediction.dim() == 2:
        prediction, target = prediction.unsqueeze(0), target.unsqueeze(0)

    T, H, W = prediction.shape
    pred_idx = _topk_indices(prediction, k)
    true_idx = _topk_indices(target,    k)

    all_dists = []
    for p, t in zip(pred_idx, true_idx):
        hits = p[torch.isin(p, t)]                                # correct cells
        if hits.numel() < 2:
            continue
        ys = (hits // W).cpu().numpy()
        xs = (hits %  W).cpu().numpy()
        coords = np.stack([ys, xs], axis=1)
        diffs = coords[:, None, :] - coords[None, :, :]
        d = np.sqrt((diffs ** 2).sum(axis=2))
        # take the strict upper triangle to avoid double-counting and self-pairs
        iu = np.triu_indices_from(d, k=1)
        all_dists.append(d[iu])

    if not all_dists:
        return np.zeros_like(distances, dtype=float)

    flat = np.concatenate(all_dists)
    return np.array([(flat <= d).mean() for d in distances])


# ---------------------------------------------------------------------------
# Convenience: full Table-I row from a (probs, targets) pair.
# ---------------------------------------------------------------------------
def evaluate_all_K(prediction: torch.Tensor,
                   target: torch.Tensor,
                   ks: Tuple[int, ...] = (5, 10, 30, 50, 100)
                   ) -> dict:
    """Compute Priority@K and NDCG@K for the K values reported in Table I."""
    out = {"priority_at_k": {}, "ndcg_at_k": {}}
    for k in ks:
        out["priority_at_k"][k] = priority_at_k(prediction, target, k)
        out["ndcg_at_k"][k]     = ndcg_at_k(prediction, target, k)
    return out
