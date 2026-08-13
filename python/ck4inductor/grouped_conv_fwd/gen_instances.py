# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
from dataclasses import replace
from functools import lru_cache
from typing import List

from ..util import library_path

from .op import CKGroupedConvFwdOp

log = logging.getLogger(__name__)


def _ck_conv_instances_path():
    conv_instances_path = os.path.join(  # noqa: F821
        library_path(),
        "include",
        "ck",
        "library",
        "tensor_operation_instance",
        "gpu",
        "grouped_conv_fwd",
    )
    if not os.path.exists(conv_instances_path):
        log.error(
            "CK library conv instances path %s does not exist", conv_instances_path
        )
        return None
    return conv_instances_path


def parse_instances(
    str_instances: List[str],
    class_name: str = "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
) -> List[CKGroupedConvFwdOp]:
    """
    Parse the lines containing Grouped Convolution Forward template instances
    into `CKGroupedConvFwdOp` instances

    `class_name` selects the device op whose template arguments are being read.
    The XDL and WMMA ops share the first 46 parameters positionally, which is all
    the shipped bias-less WMMA instances write; they diverge only in the trailing
    optionals (WMMA inserts `UseThreadTileTransfer` where XDL has
    `AComputeDataType`, and has no `LargeTensors`). Instances that write into that
    tail would be mis-parsed, so callers must not point this at a header whose
    instances do.
    """

    def maybe_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    op_instances = []
    # TODO: maybe use libclang for parsing C++ code in the future
    # to avoid this hacky parsing logic below ? :) - copilot
    for line in str_instances:
        s_template_args = line.split(class_name)[-1].strip("<>, ")
        template_args = []
        i_current = 0
        while i_current < len(s_template_args):
            if s_template_args[i_current] == " ":
                # skip whitespace
                i_current += 1
                continue
            elif s_template_args[i_current : i_current + 2] == "S<":
                # parse template S<Index...>
                i_next = s_template_args.find(">", i_current)
                template_args.append(
                    tuple(map(int, s_template_args[i_current + 2 : i_next].split(",")))
                )
                i_current = i_next + 2
            else:
                # all string attributes must be either type aliases or global constants in C++
                i_next = s_template_args.find(",", i_current)
                template_args.append(
                    maybe_int(
                        s_template_args[i_current : i_next if i_next != -1 else None]
                    )
                )
                if i_next != -1:
                    i_current = i_next + 1
            if i_next == -1:
                break

        # Mirror the two sibling scrapers: a line that does not yield a usable
        # argument list (grep -in also matches comments and forward declarations,
        # and CK may rename or reorder a template parameter) is skipped rather
        # than raised, so one upstream change cannot take down autotune for every
        # instance. The index assignments must be inside the try -- a short list
        # raises IndexError, not TypeError, which would otherwise escape and
        # defeat the purpose of this guard.
        try:
            template_args[0] = -1  # n_dim_spatial
            template_args[3] = tuple()  # ds_layout
            template_args[9] = tuple()  # ds_element_dtype

            new_instance = CKGroupedConvFwdOp(
                *template_args,  # type: ignore[arg-type]
            )
            op_instances.append(new_instance)
        except (TypeError, IndexError) as e:
            log.debug(f"{e} when parsing {line}")
    return op_instances


def _substitute_templated_args(
    op_instances: List[CKGroupedConvFwdOp],
) -> List[CKGroupedConvFwdOp]:
    """Expand the placeholder template arguments each instance leaves open.

    The shipped instances are templates: they write `BlkGemmPipeSched` / `ConvSpec`
    tokens and leave the layouts as parameters. Expand each open dimension over its
    domain, and pin the ones PyTorch's conv lowering always uses (2D, GKYXC weights,
    MNKPadding).
    """
    schedulers = [
        "BlockGemmPipelineScheduler::Intrawave",
        "BlockGemmPipelineScheduler::Interwave",
    ]
    conv_specs = [
        "ConvolutionForwardSpecialization::Default",
        "ConvolutionForwardSpecialization::Filter1x1Pad0",
        "ConvolutionForwardSpecialization::Filter1x1Stride1Pad0",
        "ConvolutionForwardSpecialization::OddC",
    ]

    substitute_instances = []
    for instance in op_instances:
        sub_scheduler = instance.block_gemm_pipeline_scheduler == "BlkGemmPipeSched"
        sub_spec = instance.conv_forward_specialization == "ConvSpec"
        schedulers_range = (
            schedulers if sub_scheduler else [instance.block_gemm_pipeline_scheduler]
        )
        spec_range = conv_specs if sub_spec else [instance.conv_forward_specialization]
        for scheduler in schedulers_range:
            for spec in spec_range:
                for channels_last in [True, False]:
                    if channels_last:
                        a_layout = "NHWGC"
                        e_layout = "NHWGK"
                    else:
                        a_layout = "NGCHW"
                        e_layout = "NGKHW"
                    substitute_instances.append(
                        replace(
                            instance,
                            block_gemm_pipeline_scheduler=scheduler,
                            conv_forward_specialization=spec,
                            gemm_specialization="GemmSpecialization::MNKPadding",
                            n_dim_spatial=2,
                            a_layout=a_layout,
                            b_layout="GKYXC",
                            e_layout=e_layout,
                        )
                    )

    return substitute_instances


@lru_cache(None)
def gen_conv_ops_library() -> List[CKGroupedConvFwdOp]:
    """
    Parse the Grouped Convolution Forward instances
    defined in the Composable Kernel library folder.
    """
    ck_library_dir = _ck_conv_instances_path()
    if not ck_library_dir:
        return []

    grep_result = subprocess.run(
        [
            "grep",
            "-inR",
            "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3",
            ck_library_dir,
        ],
        capture_output=True,
        text=True,
    )

    op_instances = parse_instances(grep_result.stdout.strip().split("\n"))

    log.debug("ck instances from library: %d", len(op_instances))

    return _substitute_templated_args(op_instances)


# Only the bias-less WMMA header is enumerated. The other WMMA headers are
# epilogue-fusion families needing a non-empty Ds tuple, which PyTorch's conv
# lowering never supplies; `..._wave_transfer_instance.hpp` is Ds-free but writes
# into the diverging tail (see parse_instances) and so cannot share this parser.
_WMMA_INSTANCE_HEADER = "device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp"


@lru_cache(None)
def gen_conv_ops_library_wmma() -> List[CKGroupedConvFwdOp]:
    """
    Parse the gfx1250 WMMA Grouped Convolution Forward instances.

    These are a separate product from the XDL instances above: purpose-built
    16x16 warp-tile kernels rather than XDL shapes that happen to lower to WMMA.
    f16/bf16 only -- CK ships no f32 WMMA conv instance at this time.
    """
    ck_library_dir = _ck_conv_instances_path()
    if not ck_library_dir:
        return []

    grep_result = subprocess.run(
        [
            "grep",
            "-in",
            "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3",
            os.path.join(ck_library_dir, _WMMA_INSTANCE_HEADER),
        ],
        capture_output=True,
        text=True,
    )

    op_instances = parse_instances(
        grep_result.stdout.strip().split("\n"),
        class_name="DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3",
    )
    # Guard the dtype rather than trusting the filename: a header rename or an
    # added dtype would otherwise let extra dtypes through unnoticed.
    op_instances = [
        op
        for op in op_instances
        if op.a_element_dtype in ("F16", "BF16")
        and op.b_element_dtype in ("F16", "BF16")
    ]

    log.debug("ck wmma conv instances from library: %d", len(op_instances))

    return [
        replace(instance, is_wmma=True)
        for instance in _substitute_templated_args(op_instances)
    ]


if __name__ == "__main__":
    print(gen_conv_ops_library())
