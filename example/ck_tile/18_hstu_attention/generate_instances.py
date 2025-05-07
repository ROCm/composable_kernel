# noqa: C801
# Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from pathlib import Path
from typing import List

HSTU_COPYRIGHT_HEADER = """
/*
  Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * The file is automatically generated, don't modify!
 * See the generator script
 * `{file}`
 */
""".format(
    file=os.path.relpath(os.path.realpath(__file__), start=Path(__file__).parents[4])
)

HSTU_FORWARD_INSTANCE_TEMPLATE_INC = """
#include <ck_tile/core/numeric/{dtype_file}.hpp>
#include \"hstu_attention_{mode}_forward_dispatch.hpp\"
"""

HSTU_FORWARD_INSTANCE_TEMPLATE = """
{extern}template void run_{mode}_forward_causal_local_bias_dropout_dispatch<
    {dtype},
    {has_causal},
    {has_local},
    {has_bias},
    {has_dropout},
    {max_k}>(HstuAttentionFwdParams& param, hipStream_t stream);
"""

HSTU_FORWARD_INSTANCE_FNAME = (
    "hstu_attention_{mode}_forward_{dtype_str}_{has_or_no_causal_str}_{has_or_no_local_str}_"
    "{has_or_no_bias_str}_{has_or_no_dropout_str}_{max_k_str}.cpp"
)

HSTU_INSTANCE_REF_FNAME = "hstu_attention_{mode}_{function}_{dtype}_instances_ref.hpp"

BOOL_MAP = {True: "true", False: "false"}

BOOL_MAP_CAUSAL = {
    True: "has_causal",
    False: "no_causal",
}

BOOL_MAP_LOCAL = {
    True: "has_local",
    False: "no_local",
}

BOOL_MAP_BIAS = {
    True: "has_bias",
    False: "no_bias",
}

BOOL_MAP_DROPOUT = {
    True: "has_dropout",
    False: "no_dropout",
}

INT_MAP_MAX_K = {hd: f"maxk_{hd}" for hd in [64, 128, 256]}

TYPE_CTYPE_MAP = {
    "fp16": "ck_tile::fp16_t",
    "bf16": "ck_tile::bf16_t",
}

TYPE_FNAME_MAP = {
    "fp16": "bfloat16",
    "bf16": "half",
}

MODE_NAME_MAP = {
    "batched": "Batched",
    "jagged": "Jagged",
}

def create_forward_instances(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "jagged"]:
        for dtype in ["fp16", "bf16"]:
            for has_causal, has_local in zip([True, False],[True, False]):
                for has_bias in [True, False]:
                    for has_dropout in [True, False]:
                        for max_k in headdims:
                            fname = HSTU_FORWARD_INSTANCE_FNAME.format(
                                mode=mode,
                                dtype_str=dtype,
                                has_or_no_causal_str=BOOL_MAP_CAUSAL[has_causal],
                                has_or_no_local_str=BOOL_MAP_LOCAL[has_local],
                                has_or_no_bias_str=BOOL_MAP_BIAS[has_bias],
                                has_or_no_dropout_str=BOOL_MAP_DROPOUT[has_dropout],
                                max_k_str=INT_MAP_MAX_K[max_k],
                            )
                            forward_instance_inc = (
                                HSTU_FORWARD_INSTANCE_TEMPLATE_INC.format(
                                    mode=mode,
                                    dtype_file=TYPE_FNAME_MAP[dtype],
                                )
                            )
                            forward_instance = HSTU_FORWARD_INSTANCE_TEMPLATE.format(
                                extern="",
                                mode=mode,
                                dtype=TYPE_CTYPE_MAP[dtype],
                                has_causal=BOOL_MAP[has_causal],
                                has_local=BOOL_MAP[has_causal],
                                has_bias=BOOL_MAP[has_bias],
                                has_dropout=BOOL_MAP[has_dropout],
                                max_k=max_k,
                                cap_mode=MODE_NAME_MAP[mode],
                            )
                            (instance_dir / fname).write_text(
                                HSTU_COPYRIGHT_HEADER
                                + forward_instance_inc
                                + forward_instance
                            )


def create_forward_instances_ref(instance_dir: Path, headdims: List) -> None:
    for mode in ["batched", "jagged"]:
        for dtype in ["fp16", "bf16"]:
            ref_fname = HSTU_INSTANCE_REF_FNAME.format(
                mode=mode,
                function="forward",
                dtype=dtype,
            )
            ref_fname_path = instance_dir / ref_fname
            forward_instance_inc = HSTU_FORWARD_INSTANCE_TEMPLATE_INC.format(
                mode=mode,
                dtype_file=TYPE_FNAME_MAP[dtype],
            )
            with open(ref_fname_path, "a") as file:
                file.write(HSTU_COPYRIGHT_HEADER)
                file.write(forward_instance_inc)
                for max_k in headdims:
                    for has_bias in [True, False]:
                        for has_dropout in [True, False]:
                            for has_causal, has_local in zip([True, False],[True, False]):
                                forward_instance = (
                                    HSTU_FORWARD_INSTANCE_TEMPLATE.format(
                                        extern="extern ",
                                        mode=mode,
                                        dtype=TYPE_CTYPE_MAP[dtype],
                                        has_causal=BOOL_MAP[has_causal],
                                        has_local=BOOL_MAP[has_local],
                                        has_bias=BOOL_MAP[has_bias],
                                        has_dropout=BOOL_MAP[has_dropout],
                                        max_k=max_k,
                                        cap_mode=MODE_NAME_MAP[mode],
                                    )
                                )
                                file.write(forward_instance)

if __name__ == "__main__":
    headdims_fwd = [64, 128, 256]

    this_dir = os.path.dirname(__file__)
    output_dir = Path(this_dir) / "instances"
    output_dir.mkdir(parents=True, exist_ok=True)

    # remove existing files in the directory
    files = os.listdir(output_dir)
    for ff in files:
        file_path = os.path.join(output_dir, ff)
        os.remove(file_path)

    create_forward_instances(output_dir, headdims_fwd)
    create_forward_instances_ref(output_dir, headdims_fwd)
