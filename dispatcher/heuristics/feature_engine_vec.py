#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Universal-GEMM feature engine extended with fixed A/B/C vector widths.

Adds six features to ``GemmUniversalFeatureEngine``::

    vec_a/b/c         fixed global vector width per operand
    vec_frac_a/b/c    effective width / widest width the problem allows

The effective width is the fixed width, or for 0 the kernel's native width from
``codegen_common.gemm_native_vector_sizes`` (tile, warps per block, warp tile,
epilogue, arch). ``vec_frac`` is not clamped and exceeds 1.0 when the effective
width is wider than the problem's legal width.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engine import GemmUniversalFeatureEngine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "codegen"))

from codegen_common import (
    _VEC_ELEMENT_BYTES,
    CommonTypeMappings,
    gemm_contiguous_dims,
    gemm_native_vector_sizes,
    gemm_problem_vector_sizes,
    normalize_gfx_arch,
)


def _dtype_max_width(dtype: str) -> int:
    """Widest vector width, in elements, for a dtype."""
    return 16 // _VEC_ELEMENT_BYTES[dtype]


def _out_dtype(dtype: str) -> str:
    """The C operand's dtype: fp8/bf8 -> fp16, int8 -> int32."""
    return CommonTypeMappings.get_output_dtype(dtype)


VEC_FEATURES = [
    "vec_a",
    "vec_b",
    "vec_c",
    "vec_frac_a",
    "vec_frac_b",
    "vec_frac_c",
]

#: Accepted layouts; the same four as the base engine's LAYOUT_MAP.
_LAYOUTS = frozenset({"rcr", "rrr", "crr", "ccr"})


def _width(value) -> int:
    """A stored width as an int. Null and negative values raise."""
    if value is None or (isinstance(value, str) and value == "") or pd.isna(value):
        raise ValueError(
            "vector width is missing; native must be encoded as 0, not as "
            f"null (got {value!r}). Only data_pipeline.parse_kernel_name "
            "populates vec_a/b/c."
        )
    width = int(value)
    if width < 0:
        raise ValueError(f"vector width must be >= 0 (0 means native), got {value!r}")
    return width


def _vec_widths(problem: dict, kernel: dict) -> list[int]:
    """The three fixed widths, kernel dict first. Absent keys raise."""
    out = []
    for operand in ("a", "b", "c"):
        key = "vec_" + operand
        for src in (kernel, problem):
            if key in src:
                out.append(_width(src[key]))
                break
        else:
            raise KeyError(
                f"{key!r} absent from both the problem and the kernel config; "
                f"{', '.join(VEC_FEATURES[:3])} are required by this engine. "
                "Pass 0 to mean native (this kernel's own width; see "
                "_native_widths)."
            )
    return out


def _extent(problem: dict, kernel: dict, lower: str) -> int:
    """One extent as ``m`` or ``M``, problem dict first. Missing or < 1 raises."""
    for src in (problem, kernel):
        for key in (lower, lower.upper()):
            if key in src and src[key] is not None:
                extent = int(src[key])
                if extent < 1:
                    raise ValueError(
                        f"extent {lower!r} must be >= 1, got {extent}; "
                        "a zero extent makes every vector width look legal"
                    )
                return extent
    raise KeyError(
        f"missing extent {lower!r}/{lower.upper()!r} in problem or kernel; "
        "cannot compute legal vector widths without it"
    )


def _required(problem: dict, kernel: dict, key: str) -> str:
    """A problem-describing key, problem dict first. Absent raises."""
    for src in (problem, kernel):
        if src.get(key) is not None:
            return str(src[key])
    raise KeyError(
        f"{key!r} is required to compute vector-width features and is absent "
        "from both the problem and the kernel config"
    )


#: Geometry keys in gemm_native_vector_sizes order: tile, warps per block
#: (codegen's ``waves``), warp tile (codegen's ``warp_tile``).
_GEOMETRY_KEYS = (
    ("tile_m", "tile_n", "tile_k"),
    ("warp_m", "warp_n", "warp_k"),
    ("warp_tile_m", "warp_tile_n", "warp_tile_k"),
)


#: Epilogues whose C store width is modelled.
_EPILOGUES = frozenset({"default", "tdm", "cshuffle"})

#: Accepted targets after suffix stripping: gfx9 plus two characters, or a
#: four-digit gfx10/11/12.
_ARCH_RE = re.compile(r"^(gfx9\d[0-9a-z]|gfx(?:10|11|12)\d{2})$")


def _check_arch(arch: str) -> str:
    normalized = normalize_gfx_arch(arch)
    if not _ARCH_RE.match(normalized):
        raise ValueError(
            f"unrecognised architecture {arch!r}; native vector widths depend on "
            "the wavefront size, which is selected by the gfx family, so an "
            "unrecognised value would silently yield wrong widths. Expected a "
            "target like gfx950, gfx942, gfx90a or gfx1250."
        )
    return normalized


def _check_epilogue(epilogue: str) -> str:
    if epilogue not in _EPILOGUES:
        raise ValueError(
            f"unrecognised epilogue {epilogue!r}; expected one of "
            f"{sorted(_EPILOGUES)}. An unmodelled epilogue would be treated as "
            "cshuffle and report a wider C store than it performs."
        )
    return epilogue


def _native_widths(dtype: str, layout: str, arch: str, epilogue: str, geom) -> tuple:
    """The A/B/C widths the kernel uses when it fixes none."""
    tile, waves, warp_tile = geom
    return gemm_native_vector_sizes(
        dtype_a=dtype,
        dtype_b=dtype,
        dtype_c=_out_dtype(dtype),
        layout=layout,
        tile=tile,
        waves=waves,
        warp_tile=warp_tile,
        gfx_arch=_check_arch(arch),
        epilogue=_check_epilogue(epilogue),
    )


def _effective_widths(raw, native, legal) -> list:
    """The three raw widths, then effective width over legal width per operand."""
    out = [float(v) for v in raw]
    for v, nat, lg in zip(raw, native, legal):
        out.append(float(nat if v == 0 else v) / float(lg))
    return out


def _geom_value(problem: dict, kernel: dict, key: str) -> int:
    """A geometry key, kernel dict first. Absent raises."""
    for src in (kernel, problem):
        if src.get(key) is not None:
            return int(src[key])
    raise KeyError(
        f"{key!r} is required to resolve this kernel's native vector widths "
        "and is absent from both the problem and the kernel config"
    )


def _kernel_key(problem: dict, kernel: dict, key: str) -> str:
    """A kernel-describing key, kernel dict first. Absent raises."""
    for src in (kernel, problem):
        if src.get(key) is not None:
            return str(src[key])
    raise KeyError(
        f"{key!r} is required to compute vector-width features and is absent "
        "from both the problem and the kernel config"
    )


def _check_layout(layout: str) -> None:
    if layout not in _LAYOUTS:
        raise ValueError(
            f"unknown layout {layout!r}; expected one of {sorted(_LAYOUTS)}. "
            "Column-major-C layouts need LAYOUT_MAP and parse_kernel_name "
            "widened first; see the _LAYOUTS comment."
        )


def _legal_widths(
    m: int, n: int, k: int, layout: str, dtype: str
) -> tuple[int, int, int]:
    """Widest legal A/B/C widths for the problem."""
    _check_layout(layout)
    return gemm_problem_vector_sizes(m, n, k, layout, dtype, dtype, _out_dtype(dtype))


class GemmUniversalVecFeatureEngine(GemmUniversalFeatureEngine):
    """``GemmUniversalFeatureEngine`` plus ``VEC_FEATURES``."""

    ENCODING_VERSION = 1

    #: Base feature count. ``super()`` results are sliced to it because the
    #: base ``extract_batch`` sizes its output from the overridden
    #: ``get_feature_names()``.
    _N_BASE = len(GemmUniversalFeatureEngine().get_feature_names())

    def get_feature_names(self):
        return GemmUniversalFeatureEngine.get_feature_names(self) + VEC_FEATURES

    def _vec_row(self, problem: dict, kernel: dict) -> list[float]:
        layout = _required(problem, kernel, "layout")
        dtype = _required(problem, kernel, "dtype")
        legal = _legal_widths(
            _extent(problem, kernel, "m"),
            _extent(problem, kernel, "n"),
            _extent(problem, kernel, "k"),
            layout,
            dtype,
        )
        raw = _vec_widths(problem, kernel)
        native = _native_widths(
            dtype,
            layout,
            _required(problem, kernel, "arch"),
            _kernel_key(problem, kernel, "epilogue"),
            tuple(
                tuple(_geom_value(problem, kernel, k) for k in group)
                for group in _GEOMETRY_KEYS
            ),
        )
        return _effective_widths(raw, native, legal)

    def extract(self, problem: dict, kernel: dict) -> np.ndarray:
        base = super().extract(problem, kernel)[: self._N_BASE]
        return np.concatenate(
            [base, np.array(self._vec_row(problem, kernel), dtype=np.float64)]
        )

    #: Columns extract_batch requires, non-null.
    _REQUIRED = (
        "m",
        "n",
        "k",
        "layout",
        "dtype",
        "arch",
        "epilogue",
        "vec_a",
        "vec_b",
        "vec_c",
        *(k for group in _GEOMETRY_KEYS for k in group),
    )

    def _native_width_columns(self, df: pd.DataFrame) -> np.ndarray:
        """Per-row native widths, resolved once per distinct kernel geometry."""
        keys = [
            "layout",
            "dtype",
            "arch",
            "epilogue",
            *(k for g in _GEOMETRY_KEYS for k in g),
        ]
        combos = df[keys].astype({k: np.int64 for g in _GEOMETRY_KEYS for k in g})
        codes, uniques = pd.factorize(
            pd.MultiIndex.from_frame(combos), use_na_sentinel=False
        )
        resolved = np.empty((len(uniques), 3), dtype=np.int64)
        for j, row in enumerate(uniques):
            lay, dt, arch, epi = row[0], row[1], row[2], row[3]
            geom = tuple(
                tuple(int(row[4 + 3 * g + o]) for o in range(3)) for g in range(3)
            )
            resolved[j] = _native_widths(str(dt), str(lay), str(arch), str(epi), geom)
        return resolved[codes]

    def extract_batch(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in ("vec_a", "vec_b", "vec_c") if c not in df.columns]
        if missing:
            raise KeyError(
                f"{missing} absent; {type(self).__name__} needs the fixed vector "
                "widths. Only data_pipeline.parse_kernel_name populates them, "
                "from the kernel name's _vec suffix. Use "
                "GemmUniversalFeatureEngine on data with no width information."
            )
        for col in self._REQUIRED:
            if col not in df.columns:
                raise KeyError(f"{col!r} is required to compute vector-width features")
            if df[col].isna().any():
                raise ValueError(
                    f"column {col!r} contains null; cannot derive legal widths. "
                    "data_pipeline backfills columns it did not populate with "
                    "None, so a null width means the kernel name was never "
                    "parsed -- native is written as 0."
                )

        layout = df["layout"].astype(str)
        unknown = sorted(set(layout) - _LAYOUTS)
        if unknown:
            raise ValueError(
                f"unknown layout(s) {unknown}; expected {sorted(_LAYOUTS)}"
            )
        if (df[["m", "n", "k"]].to_numpy(dtype=np.int64) < 1).any():
            raise ValueError(
                "extents must be >= 1; a zero extent makes every width look legal"
            )

        dtypes = df["dtype"].astype(str)
        caps_ab = np.array([_dtype_max_width(d) for d in dtypes], dtype=np.int64)
        caps_c = np.array(
            [_dtype_max_width(_out_dtype(d)) for d in dtypes], dtype=np.int64
        )
        operand_caps = (caps_ab, caps_ab, caps_c)
        dims = {d: df[d].to_numpy(dtype=np.int64) for d in ("m", "n", "k")}

        native = self._native_width_columns(df)

        extra = np.empty((len(df), len(VEC_FEATURES)), dtype=np.float64)
        for i, operand in enumerate(("a", "b", "c")):
            which = layout.map(lambda s, i=i: gemm_contiguous_dims(s)[i]).to_numpy()
            extent = np.select(
                [which == "m", which == "n", which == "k"],
                [dims["m"], dims["n"], dims["k"]],
                default=-1,
            )
            if (extent < 0).any():
                raise AssertionError(
                    "gemm_contiguous_dims returned a dim outside m/n/k"
                )
            legal = np.gcd(extent, operand_caps[i])

            raw = np.array([_width(v) for v in df["vec_" + operand]], dtype=np.int64)
            eff = np.where(raw == 0, native[:, i], raw)
            extra[:, i] = raw
            extra[:, 3 + i] = eff / legal

        base = super().extract_batch(df)[:, : self._N_BASE]
        return np.hstack([base, extra])
