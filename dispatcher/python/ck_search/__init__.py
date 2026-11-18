# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Ask/tell search strategies over a listed kernel space.

A minimal vendored copy of the ROCKE tuning search core (origin, file list and
local patches: VENDOR.md). Strategies propose trials (``ask``) and learn from
observations (``tell``); they never build or measure anything themselves.
Importing this package needs only the standard library and NumPy.

    space = SearchSpace.from_configs(records, name="512x512x512")
    strategy = make_strategy("portfolio", space, seed=0, warm_records=records[:2])
    trials = strategy.ask(4)
    strategy.tell([Observation(t, space.key(t.genome), Status.OK, value=ms)])
"""

from .factory import STRATEGY_NAMES, make_strategy
from .params import Param, SearchSpace
from .portfolio import WarmStart
from .stats import NoiseModel
from .surrogate import TablePrior
from .trial import Fidelity, Observation, Status, Trial

# Upstream commit the vendored files were copied from (see VENDOR.md).
ORIGIN_SHA = "61501dd46f68c196f16cfe3282bd979dbd0746ae"

__all__ = [
    "ORIGIN_SHA",
    "STRATEGY_NAMES",
    "Fidelity",
    "NoiseModel",
    "Observation",
    "Param",
    "SearchSpace",
    "Status",
    "TablePrior",
    "Trial",
    "WarmStart",
    "make_strategy",
]
