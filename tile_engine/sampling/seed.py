# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import datetime


def daily_seed(date=None, extra=""):
    """Return sha256(YYYY-MM-DD[:extra]) & 0xFFFFFFFF."""
    if date is None:
        date = datetime.date.today()
    material = date.isoformat()
    if extra:
        material += f":{extra}"
    return int(hashlib.sha256(material.encode()).hexdigest(), 16) & 0xFFFFFFFF


def _normalize_gpu_target(gpu_target):
    """Normalize a possibly semicolon-joined target list to a stable string.

    CMake passes multi-arch lists as "gfx90a;gfx942". Sort and deduplicate
    so the seed is invariant to target order and doesn't shift when a new
    target is appended mid-list.
    """
    if not gpu_target:
        return ""
    parts = [t.strip() for t in gpu_target.split(";") if t.strip()]
    return ";".join(sorted(set(parts)))


def make_seed(explicit_seed=None, gpu_target="", datatype="", layout=""):
    """If explicit_seed is given, return it. Otherwise compute daily seed
    with gpu_target:datatype:layout as extra material."""
    if explicit_seed is not None:
        return explicit_seed
    extra = ":".join(filter(None, [_normalize_gpu_target(gpu_target), datatype, layout]))
    return daily_seed(extra=extra)
