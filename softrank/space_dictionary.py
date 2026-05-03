"""Patch indexing for the DETER-B 5 km cube.

We slide a fixed `SIZE × SIZE` window with a fixed `STRIDE`, keep only the
patches whose *inner region* sits inside the Amazon biome mask, and then
filter again per biweek so we train only on patches that contain at least
`PIXELS_PERCENTAGE` deforested cells in the target biweek.

Three structures result:

    space_dictionary
        {patch_id: {coordinates, mask}}                        — geometry only
    biweek_space_dictionary
        {biweek: {patch_local_id: {coordinates, mask}}}        — per-biweek
    qual_biweek_dict
        {global_sample_id: biweek}                             — train / val
        used by `Deterset.__getitem__`
"""

from __future__ import annotations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from . import config as C


def create_space_dictionary(amazon_mask_padded: np.ndarray,
                            size: int = C.SIZE,
                            stride: int = C.STRIDE,
                            inner_border: int = C.INNER_BORDER
                            ) -> Dict[int, dict]:
    """Sliding-window patches whose inner region intersects the biome mask."""
    padded_y = C.NORMAL_DETER_SIZE["y"] + 2 * size
    padded_x = C.NORMAL_DETER_SIZE["x"] + 2 * size

    x_steps = list(range(0, padded_x, stride))
    y_steps = list(range(0, padded_y, stride))
    x_steps.append(padded_x - size)        # ensure full coverage at right
    y_steps.append(padded_y - size)        # and bottom

    sd: Dict[int, dict] = {}
    inner_slice = slice(inner_border, -inner_border)
    pid = 0

    for y0 in y_steps:
        for x0 in x_steps:
            patch = amazon_mask_padded[y0:y0 + size, x0:x0 + size]
            if patch[inner_slice, inner_slice].sum() > 0:
                sd[pid] = {"coordinates": [y0, x0], "mask": patch}
                pid += 1
    return sd


def compute_biweek_space_dictionary(space_dictionary: Dict[int, dict],
                                    padded_deter_area: Sequence[csr_matrix],
                                    pixels_percentage: float = C.PIXELS_PERCENTAGE,
                                    size: int = C.SIZE,
                                    inner_border: int = C.INNER_BORDER
                                    ) -> Dict[int, Dict[int, dict]]:
    """Keep, per biweek, only patches with enough inner deforested pixels."""
    inner_n = (size - 2 * inner_border) ** 2
    inner_slice = slice(inner_border, -inner_border)

    out: Dict[int, Dict[int, dict]] = {}
    for t, frame in enumerate(padded_deter_area):
        out_t: Dict[int, dict] = {}
        local_id = 0
        dense = frame.toarray() if hasattr(frame, "toarray") else np.asarray(frame)
        for entry in space_dictionary.values():
            y0, x0 = entry["coordinates"]
            inner = dense[y0 + inner_border:y0 + size - inner_border,
                          x0 + inner_border:x0 + size - inner_border]
            ratio = (inner > 0).sum() / inner_n
            if ratio >= pixels_percentage:
                out_t[local_id] = entry
                local_id += 1
        out[t] = out_t
    return out


def _ranges(biweek_space_dictionary: Dict[int, Dict[int, dict]]
            ) -> Dict[int, Tuple[int, int]]:
    """Cumulative `(start, end)` global index ranges across biweeks."""
    out: Dict[int, Tuple[int, int]] = {}
    cur = 0
    for t in sorted(biweek_space_dictionary.keys()):
        n = len(biweek_space_dictionary[t])
        out[t] = (cur, cur + n)
        cur += n
    return out


def create_qual_biweek_dicts(biweek_space_dictionary: Dict[int, Dict[int, dict]]
                             ) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Build {global_sample_idx: biweek} for train and val splits."""
    ranges = _ranges(biweek_space_dictionary)

    train_first = C.M
    train_last = C.SEASONALITY_BIWEEKS - C.VAL_BIWEEKS - C.TEST_BIWEEKS  # exclusive
    val_first = train_last
    val_last = C.SEASONALITY_BIWEEKS - C.TEST_BIWEEKS                    # exclusive

    train: Dict[int, int] = {}
    val:   Dict[int, int] = {}

    for t, (s, e) in ranges.items():
        if train_first <= t < train_last:
            for g in range(s, e):
                train[g] = t
        elif val_first <= t < val_last:
            for g in range(s, e):
                val[g] = t
    return train, val
