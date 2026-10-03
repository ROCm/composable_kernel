# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Vector-width fallback shared by the GEMM full-benchmark drivers.

A kernel requires each contiguous A/B/C extent to be a multiple of its effective
vector width, which depends on the tile and epilogue. With the fallback on,
the sweep also builds kernels with narrower fixed widths, reports
how many (tile, width) combinations were rejected or failed to compile, and
pairs every problem only with the kernels whose widths divide its own. A
fixed-width variant is not paired with a problem its own native config can
already run: it moves the same bytes with more loads, so it would only add
measurements there. --tune-c-vector-width also builds narrower C widths for
every problem and keeps them there, since a narrower C store can be faster.
"""

import itertools
from dataclasses import replace

from codegen_common import (
    CommonTypeMappings,
    VECTOR_SIZE_VARIANTS,
    gemm_contiguous_dims,
    gemm_problem_vector_sizes,
    gemm_vector_size_sweep,
)


def add_vector_fallback_arg(parser):
    parser.add_argument(
        "--no-vector-fallback",
        action="store_true",
        help="Build only native-vector-width kernels. By default, problems whose "
        "contiguous A/B/C extents are not a multiple of the native width also get "
        "kernels with narrower fixed widths (gcd of extent and native width)",
    )
    parser.add_argument(
        "--tune-c-vector-width",
        action="store_true",
        help="Also build and time narrower C widths for every problem, aligned ones "
        "included, where they can beat the native kernel by a few percent. Costs "
        "more builds and measurements",
    )


def _base_key(cfg):
    """Name of the native config a fixed-width variant was expanded from."""
    return replace(
        cfg, vector_size_a=0, vector_size_b=0, vector_size_c=0,
        pad_m=False, pad_n=False, pad_k=False,
    ).name


def _tile_fits(cfg, dims):
    """Mirrors the kernel's check: an unpadded contiguous extent must be a tile multiple."""
    return all(
        getattr(cfg, f"pad_{d}") or dims[d] % getattr(cfg, f"tile_{d}") == 0
        for d in gemm_contiguous_dims(cfg.layout)
    )


class VectorFallback:
    """Per-run fallback state: problem widths, sweep kwargs and reject counts."""

    def __init__(self, problems, layout, dtype, variant, disabled=False, tune_c=False):
        out_dtype = CommonTypeMappings.get_output_dtype(dtype)
        # Per-problem widest legal A/B/C widths for the fixed-width sweep.
        self.prob_vecs = [
            gemm_problem_vector_sizes(
                int(p["M"]), int(p["N"]), int(p["K"]), layout[:3], dtype, dtype, out_dtype
            )
            for p in problems
        ]
        # Pair using the full extents: native default epilogues can exceed
        # the fixed-width sweep's 16-byte cap (e.g. column-major fp32 on gfx12).
        self.prob_extents = [
            tuple(int(p[d.upper()]) for d in gemm_contiguous_dims(layout)) for p in problems
        ]
        self.enabled = not disabled and variant in VECTOR_SIZE_VARIANTS
        self.tune_c = tune_c
        self.rejects = {}
        # An explicit native override must win over fixed widths in the JSON.
        self.expand_kwargs = {"vector_sizes": [(0, 0, 0)]} if disabled else {}
        if self.enabled:
            # Every power-of-two width <= the problem's, for misaligned tensors
            # only (and C with tune_c); the per-problem winner among them is the width's cost.
            sweep = {
                t
                for v in self.prob_vecs
                for t in gemm_vector_size_sweep(v, dtype, dtype, out_dtype, tune_c)
            }
            self.expand_kwargs = dict(
                vector_sizes=sorted({(0, 0, 0), *sweep}), rejects=self.rejects
            )
            print(f"  Vector-width fallback triples: {self.expand_kwargs['vector_sizes']}")

    def limit_base_kernels(self, configs, max_kernels):
        """First ``max_kernels`` native kernels, each with its fixed-width variants.

        The fallback sweep tries (0, 0, 0) first, so expand_sweep emits every
        native config right before its fixed-width variants; a small limit thus
        still keeps kernels for misaligned problems. Without the fallback it is
        a plain slice.
        """
        if max_kernels <= 0:
            return configs
        if not self.enabled:
            return configs[:max_kernels]
        n_native = itertools.accumulate(not any(c.vector_sizes) for c in configs)
        return [c for c, n in zip(configs, n_native) if n <= max_kernels]

    def report_rejects(self):
        for reason, n in sorted(self.rejects.items()):
            print(f"  Vector-width reject ({n}x): {reason}")

    def report_builds(self, configs, lib_paths):
        failed = [
            c.name for c, lib in zip(configs, lib_paths) if lib is None and any(c.vector_sizes)
        ]
        for name in failed:
            print(f"  Build FAILED for fixed vector widths: {name}")
        if self.enabled:
            n_vec = sum(1 for c in configs if any(c.vector_sizes))
            n_rej = sum(self.rejects.values())
            print(
                f"  Vector-width summary: {n_vec + n_rej} fixed-width kernels requested, "
                f"{n_rej} rejected before compile, "
                f"{len(failed)} failed to compile, {n_vec - len(failed)} built"
            )

    def pairs(self, problems, built_kernels):
        """Kernel indices each problem may run (all kernels when disabled).

        A fixed-width variant is dropped for problems its native config runs,
        unless ``tune_c`` is set and it narrows only the native C width.
        """
        n_redundant = 0
        if self.enabled:
            cfgs = [cfg for cfg, _ in built_kernels]
            kernel_vecs = [cfg.effective_vector_sizes for cfg in cfgs]
            fixed = [any(cfg.vector_sizes) for cfg in cfgs]
            bases = [_base_key(cfg) for cfg in cfgs]
            native_ab = {b: kv[:2] for b, f, kv in zip(bases, fixed, kernel_vecs) if not f}
            gated = [
                f and not (self.tune_c and native_ab.get(b) == kv[:2])
                for b, f, kv in zip(bases, fixed, kernel_vecs)
            ]
            pairs = []
            for prob, pv in zip(problems, self.prob_extents):
                dims = dict(m=int(prob["M"]), n=int(prob["N"]), k=int(prob["K"]))
                fits = [all(p % k == 0 for p, k in zip(pv, kv)) for kv in kernel_vecs]
                served = {
                    b for b, f, ok, cfg in zip(bases, fixed, fits, cfgs)
                    if ok and not f and _tile_fits(cfg, dims)
                }
                idx = [i for i, ok in enumerate(fits) if ok and not (gated[i] and bases[i] in served)]
                n_redundant += sum(fits) - len(idx)
                pairs.append(idx)
        else:
            pairs = [list(range(len(built_kernels)))] * len(problems)
        n_meas = sum(map(len, pairs))
        n_incompatible = len(built_kernels) * len(problems) - n_meas - n_redundant
        print(f"  Problems: {len(problems)}")
        print(
            f"  Total measurements: {n_meas} "
            f"({n_incompatible} vector-width-incompatible pairs skipped, "
            f"{n_redundant} fixed-width pairs skipped where the native kernel runs)"
        )
        for prob, pv, idx in zip(problems, self.prob_vecs, pairs):
            if not idx:
                print(
                    f"  WARNING: no built kernel supports problem "
                    f"{prob['M']}x{prob['N']}x{prob['K']} (widths {pv})"
                )
        return pairs
