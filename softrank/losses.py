"""Loss functions evaluated in the paper.

Five losses are exposed, all sharing the same call signature

    loss(prediction, target, biome[, deforest_mask | TK_value])

so they can be swapped in/out from the trainer cleanly:

    L_Focal       — Focal-R                                 (regression baseline)
    L_WMSE        — Weighted MSE                            (regression baseline)
    L_TK          — Top-K-aware MSE proxy (Eq. 1–2)         (ranking proxy)
    L_SoftRank    — SoftRank with random sampling           (Algorithm 1)
    L_SoftRank+   — SoftRank with Ground-Truth-Injected     (Algorithm 2)

All ranking losses operate on the *inner* tensor (i.e. after `inner_border`
has been cropped out by the caller) so that boundary cells do not contribute
to the loss. The trainer handles that crop.
"""

from __future__ import annotations
from typing import Iterable, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _combine_masks(masks: Union[torch.Tensor, Iterable[torch.Tensor]]) -> torch.Tensor:
    """Multiply a list of binary masks element-wise; pass through a tensor."""
    if isinstance(masks, (list, tuple)):
        if len(masks) == 0:
            raise ValueError("Empty mask list passed to loss function.")
        out = masks[0]
        for m in masks[1:]:
            out = out * m
        return out
    return masks


# ---------------------------------------------------------------------------
# 1) Focal-R  — regression baseline (paper Section VII)
# ---------------------------------------------------------------------------
class FocalR(nn.Module):
    """Focal-flavoured L1 regression loss.

    Per-cell weight: w_m = sigmoid(beta * |y_m - ŷ_m|) ** gamma
    """

    def __init__(self, beta: float = 50.0, gamma: float = 3.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                biome: torch.Tensor) -> torch.Tensor:
        l1 = (prediction - target).abs()
        biome_mask = _combine_masks(biome)

        focal_w = torch.sigmoid(self.beta * l1).pow(self.gamma)
        weighted = focal_w * l1 * biome_mask

        if self.reduction == "mean":
            return weighted.sum() / (biome_mask.sum() + 1e-8)
        if self.reduction == "sum":
            return weighted.sum()
        return weighted


# ---------------------------------------------------------------------------
# 2) Weighted MSE — regression baseline
# ---------------------------------------------------------------------------
class WeightedMSE(nn.Module):
    """MSE up-weighted on cells that are actually deforested in the target."""

    def __init__(self, weight: float = 50.0):
        super().__init__()
        self._weight = weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                biome: torch.Tensor, deforest_mask: torch.Tensor) -> torch.Tensor:
        per_cell_loss = F.mse_loss(prediction, target, reduction="none")
        biome_mask = _combine_masks(biome)
        weights = torch.where(deforest_mask == 1,
                              biome_mask * self._weight,
                              biome_mask)
        return (per_cell_loss * weights).sum() / (biome_mask.sum() + 1e-12)


# ---------------------------------------------------------------------------
# 3) L_TK  — Top-K-aware MSE proxy  (paper Eq. 1–2)
# ---------------------------------------------------------------------------
class LTKLoss(nn.Module):
    """Top-K-thresholded MSE.

    w_m = 1 + C · σ(β · (g_m − τ_K^{(t)}))           (paper Eq. 2)

    `TK_value` is the per-biweek τ_K^{(t)} threshold, precomputed from the
    *training* deter_area cube via `softrank.utils.compute_global_TK`.
    """

    def __init__(self, C: float = 749.0, beta: float = 1.0):
        super().__init__()
        self.C = C
        self.beta = beta

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                biome: torch.Tensor, TK_value: Union[torch.Tensor, float]
                ) -> torch.Tensor:
        mse = F.mse_loss(prediction, target, reduction="none")
        biome_mask = _combine_masks(biome)

        if not torch.is_tensor(TK_value):
            TK_value = torch.tensor(TK_value, device=prediction.device,
                                    dtype=prediction.dtype)
        if TK_value.ndim == 1:                       # (B,) → (B,1,1,1)
            TK_value = TK_value.view(-1, 1, 1, 1)

        weights = 1 + self.C * torch.sigmoid(self.beta * (target - TK_value))
        weighted = mse * weights * biome_mask
        return weighted.sum() / (biome_mask.sum() + 1e-8)


# ---------------------------------------------------------------------------
# 4–5) SoftRank  &  SoftRank+
# ---------------------------------------------------------------------------
def _rank_binomial(pi_ij: torch.Tensor) -> torch.Tensor:
    """Recursive Rank-Binomial distribution (paper Eq. 9).

    Input  pi_ij  : (B, N, N) — pairwise win probabilities π_{ij} = P(s_i > s_j)
    Output probs  : (B, N, N) — probs[b, j, r] = P(item j attains rank r)
    """
    B, N, _ = pi_ij.shape
    device = pi_ij.device
    probs = torch.zeros((B, N, N), device=device)
    probs[:, :, 0] = 1.0
    identity = torch.eye(N, device=device).unsqueeze(0)        # (1, N, N)

    for i in range(N):
        current_pi = pi_ij[:, i, :].unsqueeze(2)                # (B, N, 1)
        # shift along rank axis: probs_shifted[..., r] = probs[..., r-1]
        probs_shifted = torch.cat(
            [torch.zeros((B, N, 1), device=device), probs[:, :, :-1]], dim=2)
        new_probs = probs_shifted * current_pi + probs * (1 - current_pi)
        # only update j != i (i can never beat itself)
        keep = (1.0 - identity[:, i, :]).unsqueeze(2)            # (B, N, 1)
        probs = new_probs * keep + probs * (1.0 - keep)

    return probs


def _soft_ndcg_from_candidates(s_candidates: torch.Tensor,
                               y_candidates: torch.Tensor,
                               y_flat: torch.Tensor,
                               sigma_s: float,
                               k_cutoff: int) -> torch.Tensor:
    """Compute SoftNDCG@K once tournament candidates are picked.

    Implements paper Eqs. 6–14 with the operational K-truncation Eq. 12.
    Returns per-batch-item SoftNDCG (not the loss).
    """
    B, N_total = s_candidates.shape
    device = s_candidates.device

    # Eq. 8 — pairwise probabilities through the Gaussian erf.
    diff = s_candidates.unsqueeze(2) - s_candidates.unsqueeze(1)        # (B,N,N)
    sigma_diff = math.sqrt(2) * sigma_s
    pi_ij = 0.5 * (1.0 + torch.erf(diff / (sigma_diff * math.sqrt(2))))

    # Eq. 9 — rank distributions p_j(r).
    rank_dist = _rank_binomial(pi_ij)                                   # (B,N,N)

    # Eq. 12 — operationally truncated discount D_K.
    ranks = torch.arange(N_total, dtype=torch.float32, device=device)
    discounts = 1.0 / torch.log2(ranks + 2.0)
    discounts[k_cutoff:] = 0.0

    expected_discounts = torch.matmul(rank_dist, discounts)             # (B,N)
    dcg = (y_candidates * expected_discounts).sum(dim=1)                # (B,)

    # IDCG: best K positions of the *full* ground truth (paper §IV).
    ideal_gains, _ = torch.topk(y_flat, k=N_total, dim=1)
    idcg = (ideal_gains * discounts).sum(dim=1)

    return dcg / (idcg + 1e-8)                                          # (B,)


class SoftRankLoss(nn.Module):
    """SoftRank with **random** background sampling (Algorithm 1).

    L_SoftRank = 1 − E[NDCG@K]
    """

    def __init__(self, Ntop: int, Nrandom: int, sigma_s: float, K: int):
        super().__init__()
        self.Ntop = Ntop
        self.Nrandom = Nrandom
        self.sigma_s = sigma_s
        self.K = K
        self.N_total = Ntop + Nrandom

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                biome: torch.Tensor) -> torch.Tensor:
        device = prediction.device
        B = prediction.shape[0]

        s = prediction.reshape(B, -1)
        y = target.reshape(B, -1)
        b = biome.reshape(B, -1)

        # Mask out non-Amazon cells so they cannot enter the Top-N_top.
        masked_scores = s.clone()
        masked_scores[b == 0] = -1e9
        _, top_idx = torch.topk(masked_scores, k=self.Ntop, dim=1)

        # Per-sample random padding from valid biome cells.
        random_lists = []
        for bi in range(B):
            valid = (b[bi] == 1).nonzero().flatten()
            n_avail = valid.numel()
            perm = torch.randperm(n_avail, device=device)
            if n_avail >= self.Nrandom:
                sampled = valid[perm[:self.Nrandom]]
            else:
                # not enough cells → repeat-pad
                fill = torch.randint(0, n_avail, (self.Nrandom - n_avail,),
                                     device=device)
                sampled = torch.cat([valid[perm], valid[fill]])
            random_lists.append(sampled)
        rand_idx = torch.stack(random_lists)
        combined = torch.cat([top_idx, rand_idx], dim=1)                 # (B,N)

        s_cand = torch.gather(s, 1, combined)
        y_cand = torch.gather(y, 1, combined)

        soft_ndcg = _soft_ndcg_from_candidates(
            s_cand, y_cand, y, self.sigma_s, self.K)
        return 1.0 - soft_ndcg.mean()


class SoftRankPlusLoss(nn.Module):
    """SoftRank+ with **Ground-Truth-Injected** sampling (Algorithm 2).

    L_SoftRank+ = 1 − E[NDCG@K]

    Each tournament includes:
      • Top-N_top predicted cells (Hard Negatives — also covers Hard Positives
        the model already detects),
      • Top-N_top ground-truth cells (Hard Positives — paper Scenario 1),
      • random padding to reach N_total = N_top + N_random.
    Duplicates between predicted-Top and GT-Top are deduplicated. The final
    candidate set is sorted by predicted score for matrix interpretability
    (paper Fig. 3).
    """

    def __init__(self, Ntop: int, Nrandom: int, sigma_s: float, K: int):
        super().__init__()
        self.Ntop = Ntop
        self.Nrandom = Nrandom
        self.sigma_s = sigma_s
        self.K = K
        self.N_total = Ntop + Nrandom

    def forward(self, prediction: torch.Tensor, target: torch.Tensor,
                biome: torch.Tensor) -> torch.Tensor:
        device = prediction.device
        B = prediction.shape[0]

        s = prediction.reshape(B, -1)
        y = target.reshape(B, -1)
        b = biome.reshape(B, -1)

        masked_scores = s.clone()
        masked_scores[b == 0] = -1e9

        combined_lists = []
        for bi in range(B):
            s_b, y_b = masked_scores[bi], y[bi]

            # A) Predicted Top-K (Hard Negatives focus).
            _, pred_top = torch.topk(s_b, k=self.Ntop)
            # B) Ground-Truth Top-K (Hard Positives focus).
            _, true_top = torch.topk(y_b, k=self.Ntop)
            # C) Merge & dedupe → I_core.
            core = torch.cat([pred_top, true_top]).unique()

            # D) Fill remaining tournament slots from the *valid biome
            #    background not already in I_core* (paper Algorithm 2 line 6).
            need = self.N_total - core.numel()
            valid_mask = (b[bi] == 1)
            valid_mask[core] = False
            avail = valid_mask.nonzero().flatten()
            n_av = avail.numel()

            if need > 0:
                if n_av >= need:
                    perm = torch.randperm(n_av, device=device)
                    fill_idx = avail[perm[:need]]
                elif n_av > 0:
                    # have some background — use them all + repeat-pad.
                    perm = torch.randperm(n_av, device=device)
                    pad = avail[torch.randint(0, n_av, (need - n_av,), device=device)]
                    fill_idx = torch.cat([avail[perm], pad])
                else:
                    # whole valid biome already inside I_core — pad from I_core.
                    fill_idx = core[torch.randint(0, core.numel(), (need,),
                                                  device=device)]
                cand = torch.cat([core, fill_idx])
            else:
                cand = core[: self.N_total]                             # truncate

            # E) Sort tournament cells by predicted score for clean π_{ij}
            #    matrices (paper Fig. 3). Recurrence is commutative, so this
            #    only affects interpretability, not the loss value.
            _, order = torch.sort(s_b[cand], descending=True)
            combined_lists.append(cand[order])

        combined = torch.stack(combined_lists)                           # (B,N)

        s_cand = torch.gather(s, 1, combined)
        y_cand = torch.gather(y, 1, combined)

        soft_ndcg = _soft_ndcg_from_candidates(
            s_cand, y_cand, y, self.sigma_s, self.K)
        return 1.0 - soft_ndcg.mean()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
LOSS_NAMES = ("focal", "wmse", "ltk", "softrank", "softrank_plus")


def build_loss(name: str, cfg) -> nn.Module:
    """Instantiate a loss by short name using a softrank.config.Config object."""
    name = name.lower()
    if name == "focal":
        return FocalR(beta=cfg.focal_beta, gamma=cfg.focal_gamma)
    if name == "wmse":
        return WeightedMSE(weight=cfg.wmse_weight)
    if name == "ltk":
        return LTKLoss(C=cfg.ltk_C, beta=cfg.ltk_beta)
    if name == "softrank":
        return SoftRankLoss(
            Ntop=cfg.softrank_ntop,
            Nrandom=cfg.softrank_nrandom,
            sigma_s=cfg.softrank_sigma_s,
            K=cfg.K,
        )
    if name == "softrank_plus":
        return SoftRankPlusLoss(
            Ntop=cfg.softrank_plus_ntop,
            Nrandom=cfg.softrank_plus_nrandom,
            sigma_s=cfg.softrank_plus_sigma_s,
            K=cfg.K,
        )
    raise ValueError(f"Unknown loss '{name}'. Choose from {LOSS_NAMES}.")
