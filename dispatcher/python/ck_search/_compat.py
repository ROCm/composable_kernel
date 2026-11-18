# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The three names the vendored search code took from ROCKE outside the
search package (local patch 1, see VENDOR.md).

``stable_json_hash`` is upstream ``rocke/dispatch/core.py``; ``Knobs`` and
``KnobAxis`` are upstream ``rocke/dispatch/tuning/axes.py``. The definitions,
docstrings included, are copied unchanged (``tests/test_ck_search_core.py``
compares them with upstream when ``ROCKE_PY`` is set), so
``SearchSpace.fingerprint()`` hashes match upstream.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

Knobs = Tuple[Tuple[str, object], ...]


def stable_json_hash(payload: Mapping[str, Any], *, n: int = 16) -> str:
    """Stable short SHA256 over JSON-serializable dispatcher payloads."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n]


@dataclass(frozen=True)
class KnobAxis:
    """One tuning decision over kernel-spec fields; ``choices[0]`` is ``()``,
    "leave the base value".

    ``enabler`` marks knobs that can turn an otherwise-illegal setting legal
    (an LDS-saving knob that lets a larger tile fit). They must lead the axis
    order; see :func:`rocke.dispatch.tuning.walk.iter_knob_sets`.
    """

    name: str
    choices: Tuple[Knobs, ...]
    enabler: bool = False
