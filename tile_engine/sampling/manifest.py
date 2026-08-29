# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Manifest writer for chosen_instances.json.

Each tier run emits a manifest recording the selected instances, their
parameters, the sampling method, seed, and metadata for reproducibility.
"""

import hashlib
import json
import subprocess
from pathlib import Path


def _instance_hash(params):
    """Compute a 16-hex-char fingerprint of tile+trait parameters."""
    canonical = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _get_git_sha():
    """Get current git HEAD SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def write_manifest(
    instances, output_path, op, datatype, layout, gpu_target, seed, tier, sampler_method
):
    """Write chosen_instances.json manifest.

    Args:
        instances: List of parameter dicts (flat tile+trait keys).
        output_path: Directory to write the manifest into.
        op: Op name (e.g. "gemm_universal").
        datatype: Data type string (e.g. "fp16").
        layout: Layout string (e.g. "rcr").
        gpu_target: GPU target (e.g. "gfx942").
        seed: Integer seed used for sampling.
        tier: Tier name (e.g. "daily").
        sampler_method: Sampling method string (e.g. "sobol+lhs+maximin").

    Returns:
        Path to the written manifest file.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_sha = _get_git_sha()

    manifest_entries = []
    for params in instances:
        entry = {
            "instance_hash": _instance_hash(params),
            "op": op,
            "dtype": datatype,
            "layout": layout,
            "arch": gpu_target,
        }
        # Add tile parameters
        for key in [
            "tile_m",
            "tile_n",
            "tile_k",
            "warp_m",
            "warp_n",
            "warp_k",
            "warp_tile_m",
            "warp_tile_n",
            "warp_tile_k",
        ]:
            if key in params:
                entry[key] = params[key]

        # Add trait parameters
        for key in [
            "pipeline",
            "epilogue",
            "scheduler",
            "pad_m",
            "pad_n",
            "pad_k",
            "persistent",
        ]:
            if key in params:
                entry[key] = params[key]

        entry["sampler_method"] = sampler_method
        entry["seed"] = seed
        entry["tier"] = tier
        entry["git_sha"] = git_sha

        manifest_entries.append(entry)

    manifest = {
        "version": "1.0",
        "op": op,
        "dtype": datatype,
        "layout": layout,
        "arch": gpu_target,
        "seed": seed,
        "tier": tier,
        "sampler_method": sampler_method,
        "git_sha": git_sha,
        "instance_count": len(manifest_entries),
        "instances": manifest_entries,
    }

    manifest_file = output_dir / f"chosen_instances_{op}_{datatype}_{layout}.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest_file
