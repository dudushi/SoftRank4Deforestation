"""SoftRank4Deforestation — differentiable spatial Top-K ranking for dense prediction.

Reference implementation accompanying:

    Kevin Elezi, Raul Queiroz Feitosa, Felipe Ferrari, Paolo Garza,
    Rodrigo Antônio de Souza, Francisco Gilney Silva Bezerra,
    "Learning to Rank in 2D: Differentiable Spatial Prioritization
     for Deforestation Forecast", 2026.

Public API:
    config      — every hyperparameter used in the paper.
    data        — DETER-B dataset / dataloaders.
    losses      — Focal, WMSE, LTK, SoftRank, SoftRank+.
    metrics     — Priority@K, NDCG@K, Pairwise-Distance CDF.
    models      — ResUNet backbone.
    trainer     — unified training loop with per-loss dispatch.
    inference   — sliding-biweek inference over the 2025 test split.
"""

from . import config
from . import losses
from . import metrics
from . import utils

__version__ = "1.0.0"
__all__ = ["config", "losses", "metrics", "utils"]
