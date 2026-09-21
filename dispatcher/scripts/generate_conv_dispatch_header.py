#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate the conv_python_dispatch.hpp header for the Python conv library.

Reads the include_all headers to find available kernels and creates dispatch
aliases for 2D/3D x fwd/bwd_data/bwd_weight.
"""

import argparse
import re
from pathlib import Path


def find_3d_launcher(include_all_path: Path, variant_prefix: str) -> str:
    """Find first 3D launcher name from an include_all header."""
    text = include_all_path.read_text()
    pattern = rf"(grouped_conv_{variant_prefix}_\w+_3d_\w+)\.hpp"
    match = re.search(pattern, text)
    if match:
        return match.group(1) + "_Launcher"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    kdir = Path(args.kernel_dir)

    fwd_3d = find_3d_launcher(kdir / "include_all_grouped_conv_fwd_kernels.hpp", "fwd")
    bwd_data_3d = find_3d_launcher(
        kdir / "include_all_grouped_conv_bwd_data_kernels.hpp", "bwd_data"
    )
    bwd_weight_3d = find_3d_launcher(
        kdir / "include_all_grouped_conv_bwd_weight_kernels.hpp", "bwd_weight"
    )

    lines = [
        "// Auto-generated dispatch header for Python conv library",
        "#pragma once",
        "",
        "// Forward kernels",
        '#include "include_all_grouped_conv_fwd_kernels.hpp"',
        "#define CONV_FWD_2D_AVAILABLE 1",
    ]
    if fwd_3d:
        lines += [
            "#define CONV_FWD_3D_AVAILABLE 1",
            f"using ConvFwd3dLauncher = {fwd_3d};",
        ]
    lines += [
        "",
        "// Backward data kernels",
        '#include "include_all_grouped_conv_bwd_data_kernels.hpp"',
        "#define CONV_BWD_DATA_2D_AVAILABLE 1",
    ]
    if bwd_data_3d:
        lines += [
            "#define CONV_BWD_DATA_3D_AVAILABLE 1",
            f"using ConvBwdData3dLauncher = {bwd_data_3d};",
        ]
    lines += [
        "",
        "// Backward weight kernels",
        '#include "include_all_grouped_conv_bwd_weight_kernels.hpp"',
        "#define CONV_BWD_WEIGHT_2D_AVAILABLE 1",
    ]
    if bwd_weight_3d:
        lines += [
            "#define CONV_BWD_WEIGHT_3D_AVAILABLE 1",
            f"using ConvBwdWeight3dLauncher = {bwd_weight_3d};",
        ]

    # Kernel name table for Python introspection
    names = []
    if True:  # fwd 2D always present
        names.append('"fwd_2d"')
    if fwd_3d:
        names.append('"fwd_3d"')
    if True:  # bwd_data 2D
        names.append('"bwd_data_2d"')
    if bwd_data_3d:
        names.append('"bwd_data_3d"')
    if True:  # bwd_weight 2D
        names.append('"bwd_weight_2d"')
    if bwd_weight_3d:
        names.append('"bwd_weight_3d"')

    lines += [
        "",
        "// Kernel inventory for Python",
        f"static const char* CONV_KERNEL_NAMES[] = {{{', '.join(names)}}};",
        f"static const int CONV_KERNEL_COUNT = {len(names)};",
        "",
    ]

    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"Generated dispatch header: {args.output} ({len(names)} kernels)")


if __name__ == "__main__":
    main()
