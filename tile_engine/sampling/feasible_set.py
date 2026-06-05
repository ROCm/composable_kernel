# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

GEMM_AXES = [
    "tile_m",
    "tile_n",
    "tile_k",
    "warp_m",
    "warp_n",
    "warp_k",
    "warp_tile_m",
    "warp_tile_n",
    "warp_tile_k",
    "pipeline",
    "epilogue",
    "scheduler",
    "pad_m",
    "pad_n",
    "pad_k",
    "persistent",
]

GEMM_STREAMK_AXES = GEMM_AXES + ["reduction_strategy"]

CATEGORICAL_AXES = {
    "pipeline",
    "epilogue",
    "scheduler",
    "reduction_strategy",
    "pad_m",
    "pad_n",
    "pad_k",
    "persistent",
}


def normalize_axis_values(feasible_set, axes=None):
    """Compute normalization metadata for each axis.

    Returns dict mapping axis name to:
      - For numeric axes: {"type": "numeric", "min": v, "max": v, "range": v}
      - For categorical axes: {"type": "categorical", "values": sorted list, "map": value->index}
    """
    if axes is None:
        axes = GEMM_AXES

    meta = {}
    for ax in axes:
        values = [item[ax] for item in feasible_set if ax in item]
        if not values:
            continue

        if ax in CATEGORICAL_AXES:
            unique = sorted(set(str(v) for v in values))
            meta[ax] = {
                "type": "categorical",
                "values": unique,
                "map": {v: i for i, v in enumerate(unique)},
                "count": len(unique),
            }
        else:
            num_values = [float(v) for v in values]
            mn, mx = min(num_values), max(num_values)
            meta[ax] = {
                "type": "numeric",
                "min": mn,
                "max": mx,
                "range": mx - mn if mx != mn else 1.0,
            }
    return meta


def normalize_point(item, axes, meta):
    """Normalize a single point to [0, 1] per axis."""
    coords = []
    for ax in axes:
        if ax not in meta or ax not in item:
            coords.append(0.0)
            continue
        m = meta[ax]
        if m["type"] == "numeric":
            coords.append((float(item[ax]) - m["min"]) / m["range"])
        else:
            coords.append(m["map"].get(str(item[ax]), 0) / max(m["count"] - 1, 1))
    return coords
