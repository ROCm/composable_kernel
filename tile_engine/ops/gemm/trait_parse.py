# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parse Tile Engine GEMM trait-combo strings.

A trait combo is serialized as
``<pipeline>_<epilogue>_<scheduler>_<pad_m>_<pad_n>_<pad_k>[_<persistent>]``.
Most pipeline names contain no underscore, so the historical parser simply
split on ``_``. Multi-token pipeline names (``comp_async``, ``comp_tdm``,
``comp_tdm_v2``, ``comp_async_eight_waves``, ``weight_preshuffle``) break that
split, so they are matched first against a known-prefix list (longest first);
anything else falls back to the legacy split, keeping existing names parsed
exactly as before.

The CMake mirror of this helper is ``gemm_trait_parse.cmake``.
"""

import re
from typing import List, NamedTuple

# Longest names first so ``comp_tdm_v2`` is never matched as ``comp_tdm``.
MULTI_TOKEN_PIPELINES = (
    "comp_async_eight_waves",
    "comp_tdm_v2",
    "comp_async",
    "comp_tdm",
    "weight_preshuffle",
)

_MULTI_TOKEN_RE = re.compile(
    r"^(" + "|".join(re.escape(p) for p in MULTI_TOKEN_PIPELINES) + r")_"
)


def split_trait(trait: str) -> List[str]:
    """Split a trait-combo string into its fields.

    The first field is the full pipeline name (which may itself contain
    underscores); the remaining fields are split on ``_`` as before.
    """
    match = _MULTI_TOKEN_RE.match(trait)
    if match is None:
        return trait.split("_")
    pipeline = match.group(1)
    rest = trait[match.end() :]
    return [pipeline] + (rest.split("_") if rest else [])


class TraitCombo(NamedTuple):
    pipeline: str
    epilogue: str
    scheduler: str
    pad_m: bool
    pad_n: bool
    pad_k: bool
    persistent: bool


def parse_trait(trait: str) -> TraitCombo:
    """Parse a trait-combo string into a :class:`TraitCombo`.

    ``persistent`` defaults to False when the (optional) 7th field is absent.
    Raises ValueError for a string with the wrong number of fields.
    """
    parts = split_trait(trait)
    if len(parts) not in (6, 7):
        raise ValueError(
            f"Invalid trait combo {trait!r}: expected 6 or 7 fields, got {len(parts)}"
        )
    persistent = parts[6] == "True" if len(parts) == 7 else False
    return TraitCombo(
        parts[0],
        parts[1],
        parts[2],
        parts[3] == "True",
        parts[4] == "True",
        parts[5] == "True",
        persistent,
    )


def join_trait(combo: TraitCombo, with_persistent: bool = True) -> str:
    """Inverse of :func:`parse_trait` (round-trips every valid combo)."""
    fields = [
        combo.pipeline,
        combo.epilogue,
        combo.scheduler,
        str(combo.pad_m),
        str(combo.pad_n),
        str(combo.pad_k),
    ]
    if with_persistent:
        fields.append(str(combo.persistent))
    return "_".join(fields)
