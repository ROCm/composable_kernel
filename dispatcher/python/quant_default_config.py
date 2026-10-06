# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Resolve untargeted convenience configs when the build target is known.

The public quant factories also support offline kernel-name inspection. Their
gfx950 preview must not become a build target or bypass arch-dependent defaults.
Only unchanged factory results can be resolved automatically; custom configs and
explicitly targeted factories keep the normal architecture mismatch checks.
"""

from dataclasses import asdict
from functools import wraps
from inspect import signature


def deferred_arch_default(factory):
    """Keep no-target factory calls offline and record how to resolve them.

    An omitted or None gfx_arch creates a gfx950 naming preview. setup_multiple_*
    regenerates it for its explicit or detected target, including pipeline/tile
    choices. Explicit factory arguments (such as warp_tile_k) are retained and
    validated for that target, rather than silently replaced by defaults.
    """
    factory_signature = signature(factory)

    @wraps(factory)
    def wrapped(*args, **kwargs):
        bound = factory_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["gfx_arch"] is not None:
            return factory(*args, **kwargs)

        bound.arguments["gfx_arch"] = "gfx950"
        config = factory(*bound.args, **bound.kwargs)
        config._deferred_arch_default = (wrapped, dict(bound.arguments), asdict(config))
        return config

    wrapped.__doc__ = (factory.__doc__ or "") + (
        "\n\nWith gfx_arch omitted, this returns an offline gfx950 naming preview. "
        "setup_multiple_* resolves an unchanged preview for its explicit or detected "
        "build target. Pass gfx_arch explicitly before modifying config fields or "
        "when you need the final kernel name before building."
    )
    return wrapped


def resolve_default_configs(configs, arch):
    """Return configs with untouched, untargeted defaults resolved for arch.

    Do not mutate the caller's preview or discard edits to it. A modified preview
    is ambiguous: replaying its factory could lose the edit, while preserving it
    could bypass target-specific safety checks. Require an explicitly targeted
    config in that case.
    """
    resolved = []
    for config in configs:
        provenance = getattr(config, "_deferred_arch_default", None)
        if provenance is None:
            resolved.append(config)
            continue

        factory, arguments, snapshot = provenance
        if asdict(config) != snapshot:
            raise ValueError(
                "An untargeted default config was modified after construction. "
                f"Recreate it with gfx_arch={arch!r} before applying custom fields; "
                "automatic target resolution would discard those changes."
            )
        resolved.append(factory(**{**arguments, "gfx_arch": arch}))
    return resolved
