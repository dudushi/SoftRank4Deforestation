# SoftRank4Deforestation

> **Learning to Rank in 2D: Differentiable Spatial Prioritization for Deforestation Forecast**
> Kevin Elezi · Raul Queiroz Feitosa · Felipe Ferrari · Paolo Garza · Rodrigo Antônio de Souza · Francisco Gilney Silva Bezerra (2026)

Reference implementation of the **SoftRank** and **SoftRank+** losses introduced in the paper, applied to short-term deforestation forecasting on the Brazilian Amazon (DETER-B, 2018–2025).

The repository ships:

- a clean, dependency-light PyTorch package (`softrank/`),
- two CLI entry points — `scripts/train.py` (one model) and `scripts/run_experiments.py` (full Table I, one command),
- a one-liner Hugging Face download (`scripts/download_data.py`),
- an interactive tutorial notebook (`notebooks/softrank4deforestation.ipynb`) that walks through every equation of the loss on a real prediction.

Everything below uses the **paper-default hyperparameters** (Optuna optima reported in §VI-B). Reproducing Table I numerically requires no flags.

---

## Table of contents
- [1. Quick start](#1-quick-start)
- [2. What's in the box](#2-whats-in-the-box)
- [3. Reproducing Table I](#3-reproducing-table-i)
- [4. The losses, in plain English](#4-the-losses-in-plain-english)
- [5. Using SoftRank in your own project](#5-using-softrank-in-your-own-project)
- [6. Citation](#6-citation)

---

## 1. Quick start

```bash
git clone https://github.com/dudushi/SoftRank4Deforestation.git
cd SoftRank4Deforestation
pip install -r requirements.txt

# Download the DETER-B datacubes (~ ?? GB) from Hugging Face.
python -m scripts.download_data --out data/

# Train every loss in the paper (one process, sequential).
python -m scripts.run_experiments --data-root data/

# …or train just one.
python -m scripts.train --loss softrank_plus --data-root data/
```

Hardware used in the paper: a single NVIDIA GPU with 16 GB of VRAM.
Default training is 100 epochs with early stopping (patience 10) — about
≈ ?? hours per loss on that hardware. Pass `--epochs 5` for a smoke test.

---

## 2. What's in the box

```
SoftRank4Deforestation/
├── softrank/                  package
│   ├── config.py              every paper hyperparameter, in one place
│   ├── data.py                Dataset + build_dataloaders()
│   ├── space_dictionary.py    DETER-B patch indexing
│   ├── models.py              ResUNet (Zhang et al. 2018)
│   ├── losses.py              FocalR · WMSE · LTK · SoftRank · SoftRank+
│   ├── metrics.py             Priority@K · NDCG@K · Pairwise-Distance CDF
│   ├── trainer.py             unified training loop
│   ├── inference.py           sliding-biweek test-set inference
│   └── utils.py               sparse cube I/O, seeding, helpers
├── scripts/
│   ├── download_data.py       one-liner HF download
│   ├── train.py               train ONE loss
│   ├── run_experiments.py     train ALL losses → Table I
│   └── evaluate.py            recompute Table I from saved tensors
└── notebooks/
    └── softrank4deforestation.ipynb     interactive tutorial
```

The single source of truth for every constant in the paper is `softrank/config.py` — change a value there to change the experiment.

---

## 3. Reproducing Table I

### 3.1 Single-seed reproduction (one column of Table I)

```bash
python -m scripts.run_experiments --data-root data/
```

This trains all five losses sequentially, runs inference on the held-out 2025 biweeks, and prints a table of Priority@K and NDCG@K for K ∈ {5, 10, 30, 50, 100}. It also writes `results/table_I.json` and one `.pt` tensor per loss for later analysis.

### 3.2 Full paper statistics (mean ± std over 6 seeds)

```bash
python -m scripts.run_experiments --data-root data/ --seeds 0 1 2 3 4 5
```

Per-seed checkpoints land in `checkpoints/<loss>_seed<n>.pt` and per-seed inference tensors in `results/<loss>_seed<n>.pt`. The aggregated mean ± std go to `results/table_I.json`.

### 3.3 Recomputing the table from saved tensors

If you already have inference tensors and just want to re-evaluate:

```bash
python -m scripts.evaluate --results-dir results/
```

### 3.4 Expected numbers

Mean over 6 seeds, evaluated on 2025 (held out). Reproduced from the paper's Table I:

**Priority@K**

| Loss          | K=5         | K=10        | K=30        | K=50        | K=100       |
|---------------|-------------|-------------|-------------|-------------|-------------|
| Focal         | 0.022 ±0.016 | 0.031 ±0.021 | 0.039 ±0.032 | 0.044 ±0.035 | 0.053 ±0.039 |
| WMSE          | 0.029 ±0.011 | 0.037 ±0.010 | 0.047 ±0.004 | 0.055 ±0.005 | 0.061 ±0.005 |
| LTK           | 0.054 ±0.010 | 0.058 ±0.010 | 0.075 ±0.008 | 0.083 ±0.003 | 0.089 ±0.004 |
| SoftRank      | 0.053 ±0.009 | 0.062 ±0.006 | 0.082 ±0.009 | 0.082 ±0.009 | 0.079 ±0.007 |
| **SoftRank+** | **0.071 ±0.013** | **0.072 ±0.009** | **0.089 ±0.003** | **0.096 ±0.003** | **0.103 ±0.006** |

**NDCG@K**

| Loss          | K=5         | K=10        | K=30        | K=50        | K=100       |
|---------------|-------------|-------------|-------------|-------------|-------------|
| Focal         | 0.053 ±0.033 | 0.058 ±0.037 | 0.063 ±0.044 | 0.066 ±0.046 | 0.073 ±0.051 |
| WMSE          | 0.067 ±0.015 | 0.067 ±0.013 | 0.072 ±0.008 | 0.079 ±0.010 | 0.089 ±0.008 |
| LTK           | 0.101 ±0.014 | 0.109 ±0.015 | 0.110 ±0.005 | 0.117 ±0.005 | 0.129 ±0.005 |
| SoftRank      | 0.123 ±0.011 | 0.122 ±0.008 | 0.124 ±0.010 | 0.126 ±0.010 | 0.121 ±0.009 |
| **SoftRank+** | **0.130 ±0.010** | **0.130 ±0.007** | **0.134 ±0.004** | **0.139 ±0.004** | **0.147 ±0.007** |

---

## 4. The losses, in plain English

Five losses are evaluated. All share the call signature `loss(pred, target, biome, …)` and operate on the *inner* tensor (boundary cells stripped — see §VI-A of the paper).

| Name | Module | Description |
|---|---|---|
| `focal`         | `FocalR`            | Focal-flavoured L1 regression — paper baseline. |
| `wmse`          | `WeightedMSE`       | MSE up-weighted on deforested cells — paper baseline. |
| `ltk`           | `LTKLoss`           | Top-K-aware MSE proxy (Eq. 1–2). The hypothesis-test that motivated SoftRank. |
| `softrank`      | `SoftRankLoss`      | SoftNDCG@K with **random** background sampling (Algorithm 1). |
| `softrank_plus` | `SoftRankPlusLoss`  | SoftNDCG@K with **Ground-Truth-Injected** sampling (Algorithm 2 — *the paper's main contribution*). |

The interactive notebook in `notebooks/softrank4deforestation.ipynb` walks through every equation on a real prediction, including the K-truncation (Eq. 12) that makes the spatial gradient sparse and operationally meaningful.

---

## 5. Using SoftRank in your own project

The losses are self-contained — copy `softrank/losses.py` into any PyTorch project. Minimal example:

```python
from softrank.losses import SoftRankPlusLoss

loss_fn = SoftRankPlusLoss(Ntop=52, Nrandom=118, sigma_s=0.0173, K=100)

# pred, target, biome each shaped (B, H, W) or (B, 1, H, W)
loss = loss_fn(pred, target, biome)
loss.backward()
```

The framework is **task-agnostic**: anywhere you have a 2D dense prediction and only act on the Top-K cells (medical-image triage, disaster response, invasive-species control, precision agriculture), the same template applies — the *sampling heuristic* in `softrank/losses.py` is the only thing you may want to specialise.

---

## 6. Citation

If you use this code or the SoftRank+ formulation, please cite:

```bibtex
@article{elezi2026softrank,
  title   = {Learning to Rank in 2D: Differentiable Spatial Prioritization for Deforestation Forecast},
  author  = {Elezi, Kevin and Feitosa, Raul Queiroz and Ferrari, Felipe and Garza, Paolo and de Souza, Rodrigo Ant{\^o}nio and Bezerra, Francisco Gilney Silva},
  year    = {2026}
}
```

---

## Acknowledgments

DETER-B data courtesy of the Brazilian National Institute for Space Research (INPE).
The 2025 test split was built using historical observations through the end of 2024 only — no leakage.
