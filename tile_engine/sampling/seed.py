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


def make_seed(explicit_seed=None, gpu_target="", datatype="", layout=""):
    """If explicit_seed is given, return it. Otherwise compute daily seed
    with gpu_target:datatype:layout as extra material."""
    if explicit_seed is not None:
        return explicit_seed
    extra = ":".join(filter(None, [gpu_target, datatype, layout]))
    return daily_seed(extra=extra)
