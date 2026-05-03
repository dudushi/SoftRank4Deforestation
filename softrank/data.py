"""DETER-B Datasets and DataLoader factory.

This module is the single entry point users should import to get
`(train_loader, val_loader, test_loader)` ready to feed the trainer.
The actual numpy/sparse cube I/O lives in `softrank.utils`.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import rioxarray
import torch
from scipy.sparse import csr_matrix
from torch.utils.data import DataLoader, Dataset

from . import config as C
from . import utils as U
from .space_dictionary import (
    create_space_dictionary,
    compute_biweek_space_dictionary,
    create_qual_biweek_dicts,
)


# ---------------------------------------------------------------------------
# Per-sample dataset
# ---------------------------------------------------------------------------
class Deterset(Dataset):
    """Train/val DETER-B dataset.

    Each sample is a (M+M+M+(M-1)+N, SIZE, SIZE) tensor stacking historical
    alerts, counts, cumulatives, first-differences, and the seasonality
    feature for the prediction biweek; plus the SIZE × SIZE label and biome
    mask, the patch top-left coordinates, and the per-biweek τ_K threshold.
    """

    def __init__(
        self,
        deter_area: Iterable[csr_matrix],
        deter_count: Iterable[csr_matrix],
        deter_accumulated: Iterable[csr_matrix],
        deter_seasonal: Iterable[csr_matrix],
        biweek_space_dictionary: Dict[int, Dict[int, dict]],
        qual_biweek_dict: Dict[int, int],
        TKi: Dict[int, float],
        starting_idx: int,
    ) -> None:
        super().__init__()
        self.deter_area = list(deter_area)
        self.deter_count = list(deter_count)
        self.deter_accumulated = list(deter_accumulated)
        self.deter_seasonal = list(deter_seasonal)
        self.bsd = biweek_space_dictionary
        self.qual = qual_biweek_dict
        self.TKi = TKi
        self.starting = starting_idx

    def __len__(self) -> int:
        return len(self.qual)

    def __getitem__(self, idx: int):
        biweek = self.qual[self.starting + idx]
        space_idx = idx % len(self.bsd[biweek])
        y0, x0 = self.bsd[biweek][space_idx]["coordinates"]

        ys = slice(y0, y0 + C.SIZE)
        xs = slice(x0, x0 + C.SIZE)
        ts = slice(biweek - C.M, biweek)

        alerts       = np.stack([w[ys, xs].toarray() for w in self.deter_area[ts]])
        counts       = np.stack([w[ys, xs].toarray() for w in self.deter_count[ts]])
        accumulated  = np.stack([w[ys, xs].toarray() for w in self.deter_accumulated[ts]])
        diff_alerts  = np.diff(alerts, axis=0)
        seasonality  = np.stack([
            w[ys, xs].toarray()
            for w in self.deter_seasonal[biweek:biweek + C.N]
        ])

        x = np.concatenate([alerts, diff_alerts, counts, accumulated, seasonality],
                           axis=0).astype(np.float32)

        y = np.stack([w[ys, xs].toarray()
                      for w in self.deter_area[biweek:biweek + C.N]])

        biome = self.bsd[biweek][space_idx]["mask"]
        TK_value = self.TKi[biweek]

        return x, y, biome, (y0, x0), TK_value


# ---------------------------------------------------------------------------
# Inference (sliding biweek over the held-out 2025 test split)
# ---------------------------------------------------------------------------
class InferenceSet(Dataset):
    """Iterate every biweek of the test split, returning the full padded scene."""

    def __init__(
        self,
        deter_area: Iterable[csr_matrix],
        deter_count: Iterable[csr_matrix],
        deter_accumulated: Iterable[csr_matrix],
        deter_seasonal: Iterable[csr_matrix],
        amazon_mask_padded: np.ndarray,
    ) -> None:
        super().__init__()
        deter_area = list(deter_area)
        deter_count = list(deter_count)
        deter_accumulated = list(deter_accumulated)
        deter_seasonal = list(deter_seasonal)

        sl = slice(len(deter_area) - C.TEST_BIWEEKS - C.M, len(deter_area))
        self.deter_area = deter_area[sl]
        self.deter_count = deter_count[sl]
        self.deter_accumulated = deter_accumulated[sl]
        self.deter_seasonal = deter_seasonal[sl]
        self.amazon_mask = amazon_mask_padded

    def __len__(self) -> int:
        return C.TEST_BIWEEKS

    def __getitem__(self, idx: int):
        ts = slice(idx, idx + C.M)
        alerts      = np.stack([w.toarray() for w in self.deter_area[ts]])
        counts      = np.stack([w.toarray() for w in self.deter_count[ts]])
        accumulated = np.stack([w.toarray() for w in self.deter_accumulated[ts]])
        diff_alerts = np.diff(alerts, axis=0)
        seasonality = np.stack([
            w.toarray()
            for w in self.deter_seasonal[idx + C.M: idx + C.M + C.N]
        ])
        x = np.concatenate([alerts, diff_alerts, counts, accumulated, seasonality],
                           axis=0).astype(np.float32)
        y = np.stack([w.toarray()
                      for w in self.deter_area[idx + C.M: idx + C.M + C.N]])
        return x, y, self.amazon_mask


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------
def build_dataloaders(data_root: str | Path,
                      batch_size: int = C.BATCH_SIZE,
                      num_workers: int = C.NUM_WORKERS,
                      pin_memory: bool = C.PIN_MEMORY
                      ) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build (train, val, test) loaders + a side dict with cubes & TK thresholds.

    `data_root` must contain:
        deter_area.npz
        deter_count.npz
        deter_cummul.npz
        deter_seasonal.npz
        5km_biomemask.tif
        5km_biomemask_padded.tif
    See `scripts/download_data.py` for the canonical download command.
    """
    data_root = Path(data_root)
    print(f"[1/4] Reading sparse cubes from {data_root} …")

    biome_mask        = rioxarray.open_rasterio(data_root / "5km_biomemask.tif").values[0]
    biome_mask_padded = rioxarray.open_rasterio(data_root / "5km_biomemask_padded.tif").values[0]

    deter_area        = U.load_sparse_npz(str(data_root / "deter_area.npz"),
                                          start=C.SEASONALITY_BIWEEKS)
    deter_count       = U.load_sparse_npz(str(data_root / "deter_count.npz"),
                                          start=C.SEASONALITY_BIWEEKS)
    deter_accumulated = U.load_sparse_npz(str(data_root / "deter_cummul.npz"),
                                          start=C.SEASONALITY_BIWEEKS)
    deter_seasonal    = U.load_sparse_npz(str(data_root / "deter_seasonal.npz"),
                                          start=0)

    pad = C.SIZE
    print(f"[2/4] Padding cubes by {pad} px on each side …")
    p_area        = U.pad_sparse_cube(deter_area,        pad)
    p_count       = U.pad_sparse_cube(deter_count,       pad)
    p_accumulated = U.pad_sparse_cube(deter_accumulated, pad)
    p_seasonal    = U.pad_sparse_cube(deter_seasonal,    pad)

    print("[3/4] Building patch indices …")
    sd  = create_space_dictionary(biome_mask_padded)
    bsd = compute_biweek_space_dictionary(sd, p_area)
    qual_train, qual_val = create_qual_biweek_dicts(bsd)

    TKi = U.compute_global_TK(deter_area, K=C.K)

    train_set = Deterset(p_area, p_count, p_accumulated, p_seasonal,
                         bsd, qual_train, TKi,
                         starting_idx=min(qual_train.keys()))
    val_set   = Deterset(p_area, p_count, p_accumulated, p_seasonal,
                         bsd, qual_val, TKi,
                         starting_idx=max(qual_train.keys()) + 1)
    test_set  = InferenceSet(p_area, p_count, p_accumulated, p_seasonal,
                             biome_mask_padded)

    print("[4/4] Wrapping in DataLoaders …")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False,
                              num_workers=0,            pin_memory=pin_memory)

    aux = dict(biome_mask=biome_mask, biome_mask_padded=biome_mask_padded,
               deter_area=deter_area, TKi=TKi)

    print()
    print(f"  train: {len(train_set):>6}  val: {len(val_set):>6}  "
          f"test: {len(test_set):>3}  (biweeks)")
    return train_loader, val_loader, test_loader, aux
