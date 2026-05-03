"""All hyperparameters used in the paper, in one place.

Every value here is paper-verbatim — changing any of them breaks
reproducibility of Table I. The values below are the *result* of the
Optuna search reported in Section VI-B.

If you want to experiment, override fields on the returned object
rather than editing this file (e.g. `cfg = Config(); cfg.lr = 5e-4`).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 1738


# ---------------------------------------------------------------------------
# Spatial / temporal grid (DETER-B at 5 km, biweekly)
# ---------------------------------------------------------------------------
NORMAL_DETER_SIZE: Dict[str, int] = {"y": 488, "x": 677}

# Patch size and the inner border that is masked out of the loss
# (boundary cells have limited spatial context).
SIZE: int = 32
INNER_BORDER: int = 4

# Patch sampling.
STRIDE: int = 13
PIXELS_PERCENTAGE: float = 0.0008  # min deforested fraction inside a patch
                                   # to keep it in the training set.

# Temporal window: M past biweeks → predict N future biweek(s).
M: int = 8
N: int = 1

# DETER-B coverage and chronological splits.
SEASONALITY_BIWEEKS: int = 199        # length of the seasonality cube;
                                      #   we trim other cubes to align.
VAL_BIWEEKS: int = 12                 # second half of 2024
TEST_BIWEEKS: int = 24                # all of 2025 (held out)

STARTING_DATE: str = "16/09/2017"
ENDING_DATE: str = "16/12/2025"


# ---------------------------------------------------------------------------
# Feature stack (channels)
# ---------------------------------------------------------------------------
# Per biweek we stack:
#   M  alerts                (deter_area)
#   M  alert counts          (deter_count)
#   M  cumulative alerts     (deter_accumulated)
#   M-1 first-differences of alerts
#   N  seasonality forecast  (deter_seasonal)
CHANNELS: int = M + M + M + (M - 1) + N


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
EPOCHS: int = 100
BATCH_SIZE: int = 64
LR: float = 1e-3
WEIGHT_DECAY: float = 1e-5
PATIENCE: int = 10                    # early-stopping
NUM_WORKERS: int = 0
PIN_MEMORY: bool = True
DEVICE: str = "cuda"

# ResUNet capacity (matches Elezi et al. 2026 [13]).
FILTERS: List[int] = [16, 16, 16, 16]


# ---------------------------------------------------------------------------
# Operational budget K (Top-K) — the constant referenced everywhere.
# ---------------------------------------------------------------------------
K: int = 100


# ---------------------------------------------------------------------------
# Per-loss hyperparameters (frozen Optuna optima from the paper)
# ---------------------------------------------------------------------------
# Focal-R (Section VII)
FOCAL_BETA: float = 50.0
FOCAL_GAMMA: float = 3.0

# Weighted-MSE
WMSE_WEIGHT: float = 50.0

# LTK (paper §II-A): w_m = 1 + C · σ(β·(g_m − τ_K))
LTK_C: float = 749.0
LTK_BETA: float = 1.0

# SoftRank — random sampling (Algorithm 1).
SOFTRANK_SIGMA_S: float = 0.02968092964436321
SOFTRANK_NTOP: int = 67
SOFTRANK_NRANDOM: int = 60

# SoftRank+ — Ground-Truth-Injected sampling (Algorithm 2).
SOFTRANK_PLUS_SIGMA_S: float = 0.01729147580097526
SOFTRANK_PLUS_NTOP: int = 52
SOFTRANK_PLUS_NRANDOM: int = 118


# ---------------------------------------------------------------------------
# LR scheduler (ReduceLROnPlateau, mode='max' on Val NDCG@K)
# ---------------------------------------------------------------------------
# Note: the paper uses two slightly different schedulers depending on the
# loss family. We expose both and dispatch in the trainer.
SCHED_FACTOR: float = 0.72
SCHED_THRESHOLD: float = 1e-3
SCHED_MIN_LR: float = 3e-7
SCHED_PATIENCE_REGRESSION: int = 1   # for Focal / WMSE / LTK (train.py)
SCHED_PATIENCE_RANK: int = 3         # for SoftRank / SoftRank+ (train_softrank.py)


# ---------------------------------------------------------------------------
# Convenience: dataclass aggregation
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """Aggregated mutable view of all hyperparameters above."""
    # reproducibility
    seed: int = SEED

    # grid
    size: int = SIZE
    inner_border: int = INNER_BORDER
    stride: int = STRIDE
    pixels_percentage: float = PIXELS_PERCENTAGE
    M: int = M
    N: int = N
    seasonality_biweeks: int = SEASONALITY_BIWEEKS
    val_biweeks: int = VAL_BIWEEKS
    test_biweeks: int = TEST_BIWEEKS
    starting_date: str = STARTING_DATE
    ending_date: str = ENDING_DATE

    # network
    channels: int = CHANNELS
    filters: List[int] = field(default_factory=lambda: list(FILTERS))

    # optimization
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    patience: int = PATIENCE
    num_workers: int = NUM_WORKERS
    pin_memory: bool = PIN_MEMORY
    device: str = DEVICE

    # ranking
    K: int = K

    # loss-specific
    focal_beta: float = FOCAL_BETA
    focal_gamma: float = FOCAL_GAMMA
    wmse_weight: float = WMSE_WEIGHT
    ltk_C: float = LTK_C
    ltk_beta: float = LTK_BETA
    softrank_sigma_s: float = SOFTRANK_SIGMA_S
    softrank_ntop: int = SOFTRANK_NTOP
    softrank_nrandom: int = SOFTRANK_NRANDOM
    softrank_plus_sigma_s: float = SOFTRANK_PLUS_SIGMA_S
    softrank_plus_ntop: int = SOFTRANK_PLUS_NTOP
    softrank_plus_nrandom: int = SOFTRANK_PLUS_NRANDOM

    # scheduler
    sched_factor: float = SCHED_FACTOR
    sched_threshold: float = SCHED_THRESHOLD
    sched_min_lr: float = SCHED_MIN_LR
    sched_patience_regression: int = SCHED_PATIENCE_REGRESSION
    sched_patience_rank: int = SCHED_PATIENCE_RANK

    # ----- derived -----
    @property
    def padded_deter_size(self) -> Dict[str, int]:
        return {
            "y": NORMAL_DETER_SIZE["y"] + 2 * self.size,
            "x": NORMAL_DETER_SIZE["x"] + 2 * self.size,
        }

    def asdict(self) -> dict:
        return asdict(self)


def default_config() -> Config:
    """Return a fresh paper-default Config instance."""
    return Config()
