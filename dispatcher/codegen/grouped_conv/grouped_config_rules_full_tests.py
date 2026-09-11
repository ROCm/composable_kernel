# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from typing import List

def _classify_config(cfg) -> str:
    """Classify a config into a feature category for stratified test selection.

    The datatype tag is folded into the category so that the stratified subset keeps
    at least one config per datatype per feature category.
    """
    from unified_grouped_conv_codegen import (
        DepthwiseConvKernelConfig,
    )
    dt = getattr(cfg, "datatype", None) or "fp16"
    if isinstance(cfg, DepthwiseConvKernelConfig):
        return f"depthwise:{dt}"
    tr = cfg.trait
    if getattr(tr.streamk_config, "streamk_enabled", False):
        return f"streamk:{dt}"
    if tr.split_image:
        return f"split_image:{dt}"
    if tr.num_groups_to_merge > 1:
        return f"merged_groups:{dt}"
    if tr.two_stage:
        return f"two_stage:{dt}"
    if tr.explicit_gemm:
        return f"explicit_gemm:{dt}"
    return f"regular:{dt}"


def _select_test_configs(configs) -> List:
    """Select ~20% of configs with stratified sampling for test builds.

    Guarantees:
      1. At least 1 config from each feature category.
      2. Every (pipeline, scheduler) combo per variant is represented.

    Selection: every 5th config (indices 4, 9, 14, ...) from each category,
    matching awk 'NR % 5 == 0' convention.
    """
    from collections import defaultdict
    from unified_grouped_conv_codegen import (
        GroupedConvKernelConfig
    )

    # Config dataclasses are mutable (unhashable), so membership is tracked by
    # position in the original list rather than id() (which is only valid while
    # objects stay alive) or a content key (which would require hashability).
    configs = list(configs)

    categories = defaultdict(list)  # category -> list of original indices
    for idx, cfg in enumerate(configs):
        categories[_classify_config(cfg)].append(idx)

    selected = set()  # original indices

    # Take ~20% from each category (minimum 1)
    for cat, cat_indices in categories.items():
        cat_selected = False
        for rank, idx in enumerate(cat_indices):
            if (rank + 1) % 5 == 0:
                selected.add(idx)
                cat_selected = True
        # Ensure minimum 1 per category
        if not cat_selected and cat_indices:
            selected.add(cat_indices[0])

    # Ensure pipeline/scheduler coverage per (variant, datatype) (GEMM only).
    gemm_indices = [
        i for i, c in enumerate(configs)
        if isinstance(c, GroupedConvKernelConfig)
    ]
    variant_combos = defaultdict(set)
    variant_covered = defaultdict(set)
    for i in gemm_indices:
        c = configs[i]
        vkey = (c.variant, getattr(c, "datatype", None) or "fp16")
        combo = (c.trait.pipeline, c.trait.scheduler)
        variant_combos[vkey].add(combo)
        if i in selected:
            variant_covered[vkey].add(combo)

    for vkey, required in variant_combos.items():
        variant, dt = vkey
        missing = required - variant_covered[vkey]
        for combo in missing:
            for i in gemm_indices:
                c = configs[i]
                if c.variant == variant and \
                        (getattr(c, "datatype", None) or "fp16") == dt and \
                        (c.trait.pipeline, c.trait.scheduler) == combo:
                    selected.add(i)
                    break

    return [configs[i] for i in sorted(selected)]

def get_configs(
    arch: str,
    variants: List,
    ndims: List[int],
    datatypes: List[str]
  ) -> List:
    """Build all available configs for the "full-tests" rule set.

    Unified rule-set entry point used by
    ``unified_grouped_conv_codegen.get_default_configs``.
    Trims down the "full" config set using the rules defined in ``_select_test_configs``.
    """

    from .grouped_config_rules_full import get_configs as get_full_configs

    all_configs = get_full_configs(arch, variants, ndims, datatypes)
    test_configs = _select_test_configs(all_configs)
    return test_configs


def _select_tiny_configs(configs, min_count: int = 10) -> List:
    """Select a minimal set of configs for quick development builds.

    Uses the same category mechanism as ``_select_test_configs``
    (``_classify_config`` folds the feature category and datatype together, so a
    represented category also represents its variant), additionally split by
    spatial dimensionality so both 2D and 3D kernels are represented, but
    maximally trimmed down: pick a single config per (category, ndim), then
    round-robin fill up to ``min_count`` so the set is at least ``min_count``
    configs (or all available, if fewer). Every (feature category, ndim) present
    in ``configs`` is represented.
    """
    from collections import OrderedDict

    by_category: "OrderedDict[tuple, list]" = OrderedDict()
    for cfg in configs:
        key = (_classify_config(cfg), getattr(cfg, "ndim_spatial", None))
        by_category.setdefault(key, []).append(cfg)

    selected: List = []
    # One per category first so every category (and thus variant) is represented.
    for cfgs in by_category.values():
        if cfgs:
            selected.append(cfgs[0])

    # Fill up to min_count round-robin across categories.
    idx = 1
    while len(selected) < min_count:
        added = False
        for cfgs in by_category.values():
            if idx < len(cfgs):
                selected.append(cfgs[idx])
                added = True
                if len(selected) >= min_count:
                    break
        if not added:
            break  # all configs exhausted
        idx += 1

    return selected


def get_tiny_configs(
    arch: str,
    variants: List,
    ndims: List[int],
    datatypes: List[str],
) -> List:
    """Build the "tiny" rule set: a minimal subset of the "full-tests" rule set.

    Returns at least 10 configs (or all available, if fewer), with every feature
    category represented (same category mechanism as the "full-tests" rule set, but
    maximally trimmed). Intended for fast development/iteration builds.
    """
    test_configs = get_configs(arch, variants, ndims, datatypes)
    return _select_tiny_configs(test_configs)