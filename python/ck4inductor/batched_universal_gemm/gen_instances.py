# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import logging
import os
import subprocess
from dataclasses import replace
from functools import lru_cache
from typing import List

from ..util import library_path, canonical_instances

from .op import CKBatchedGemmOperation

log = logging.getLogger(__name__)


def _ck_library_dir():
    gemm_instances_path = os.path.join(
        library_path(),
        "src",
        "tensor_operation_instance",
        "gpu",
        "gemm_universal_batched",
    )
    if not os.path.exists(gemm_instances_path):
        log.error("CK library path %s does not exist", gemm_instances_path)
        return None
    return gemm_instances_path


def _ck_wmma_library_dir():
    gemm_instances_path = os.path.join(
        library_path(),
        "src",
        "tensor_operation_instance",
        "gpu",
        "batched_gemm",
    )
    if not os.path.exists(gemm_instances_path):
        log.error("CK library path %s does not exist", gemm_instances_path)
        return None
    return gemm_instances_path


def parse_instances(
    str_instances: List[str],
    class_name: str = "DeviceBatchedGemmMultiD_Xdl_CShuffle_V3",
    ds_mode: str = "overwrite",
) -> List[CKBatchedGemmOperation]:
    """
    Parse the lines containing Universal Gemm template instances into `CKBatchedGemmOperation` instances

    `class_name` is the CK device-op class the instance lines instantiate.

    `ds_mode` selects how the two Ds slots are reconciled with the dataclass, and it does NOT
    follow from `class_name` -- the two device ops genuinely differ. `DeviceBatchedGemmMultiD_*`
    declares `DsLayout`/`DsDataType` at positions 2 and 6 (44 template args), so their parsed
    placeholders are *overwritten*. `DeviceBatchedGemm_Wmma_CShuffleV3` has no Ds parameters at
    all (42 args), so empty slots must be *inserted* to line the remaining fields up. Getting
    this wrong shifts every subsequent field by two and yields plausible-looking garbage.
    """

    # Rejected explicitly: an unrecognized value would fall
    # through to the insert branch and silently shift every field by two.
    if ds_mode not in ("overwrite", "insert"):
        raise ValueError(f"ds_mode must be 'overwrite' or 'insert', got {ds_mode!r}")

    def maybe_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    op_instances = []
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

        # The Ds reconciliation has to sit inside the guard: `grep -inR` also matches
        # comments and forward declarations, which yield too few template args, and
        # subscript assignment then raises IndexError rather than TypeError. Left
        # outside, one such line would abort enumeration instead of being skipped.
        try:
            if ds_mode == "overwrite":
                # ds layout and dtype are parsed as placeholder; reset value
                template_args[2] = tuple()  # ds layout
                template_args[6] = tuple()  # ds dtype
            else:
                template_args.insert(2, tuple())  # ds layout
                template_args.insert(6, tuple())  # ds dtype

            new_instance = CKBatchedGemmOperation(
                *template_args,  # type: ignore[arg-type]
            )
            op_instances.append(new_instance)
        except (TypeError, IndexError) as e:
            log.debug(f"{e} when parsing {line}")
    return op_instances


@lru_cache(None)
def gen_ops_library() -> List[CKBatchedGemmOperation]:
    """
    Parse the Universal Gemm instances defined in the composable kernel library folder.
    """
    ck_library_dir = _ck_library_dir()
    if not ck_library_dir:
        return []

    grep_result = subprocess.run(
        [
            "grep",
            "-inR",
            "DeviceBatchedGemmMultiD_Xdl_CShuffle_V3",
            _ck_library_dir(),
        ],
        capture_output=True,
        text=True,
    )

    op_instances = parse_instances(grep_result.stdout.strip().split("\n"))

    log.debug("ck instances from library: %d", len(op_instances))

    return canonical_instances(_substitute_scheduler_spec(op_instances))


def _substitute_scheduler_spec(
    op_instances: List[CKBatchedGemmOperation],
) -> List[CKBatchedGemmOperation]:
    """Expand each parsed instance across the scheduler x GemmSpecialization domains,
    but only for the fields left as placeholders (`BlkGemmPipeSched` / `GemmSpec`)."""
    schedulers = [
        "BlockGemmPipelineScheduler::Intrawave",
        "BlockGemmPipelineScheduler::Interwave",
    ]
    gemm_specs = [
        "GemmSpecialization::Default",
        "GemmSpecialization::MPadding",
        "GemmSpecialization::NPadding",
        "GemmSpecialization::KPadding",
        "GemmSpecialization::MNPadding",
        "GemmSpecialization::MKPadding",
        "GemmSpecialization::NKPadding",
        "GemmSpecialization::MNKPadding",
    ]

    # substitute templated args by looping through their domains
    substitute_instances = []
    for instance in op_instances:
        sub_scheduler = instance.block_gemm_pipeline_scheduler == "BlkGemmPipeSched"
        sub_spec = instance.gemm_specialization == "GemmSpec"
        schedulers_range = (
            schedulers if sub_scheduler else [instance.block_gemm_pipeline_scheduler]
        )
        spec_range = gemm_specs if sub_spec else [instance.gemm_specialization]
        for scheduler in schedulers_range:
            for spec in spec_range:
                substitute_instances.append(
                    replace(
                        instance,
                        block_gemm_pipeline_scheduler=scheduler,
                        gemm_specialization=spec,
                    )
                )

    return substitute_instances


@lru_cache(None)
def gen_ops_library_wmma() -> List[CKBatchedGemmOperation]:
    """
    Parse the gfx1250 WMMA batched Gemm instances (`DeviceBatchedGemm_Wmma_CShuffleV3`) shipped
    in the composable kernel library folder. These are the 16x16-warp WMMA instances the CKWMMA
    PyTorch backend renders through `DeviceBatchedGemmMultiD_Wmma_CShuffleV3`.

    These live in a different folder than the XDL batched instances, are only present as .cpp
    sources, and omit the two Ds template parameters -- hence the separate library dir and
    `ds_mode="insert"`.
    """
    ck_library_dir = _ck_wmma_library_dir()
    if not ck_library_dir:
        return []

    grep_result = subprocess.run(
        [
            "grep",
            "-inR",
            "DeviceBatchedGemm_Wmma_CShuffleV3",
            ck_library_dir,
        ],
        capture_output=True,
        text=True,
    )

    op_instances = parse_instances(
        grep_result.stdout.strip().split("\n"),
        class_name="DeviceBatchedGemm_Wmma_CShuffleV3",
        ds_mode="insert",
    )

    # Keep only fp16/bf16 instances (a/b/c all in {F16, BF16}); WMMA also ships fp8/i4
    # variants that are out of scope for the CKWMMA bmm path.
    allowed_dtypes = {"F16", "BF16"}
    op_instances = [
        op
        for op in op_instances
        if op.a_element_dtype in allowed_dtypes
        and op.b_element_dtype in allowed_dtypes
        and op.c_element_dtype in allowed_dtypes
    ]

    # The WMMA instance sources spell the scheduler as a bare `Intrawave`/`Interwave`
    # (resolved by a file-local `static constexpr auto` in the instance source), while
    # the XDL sources use the `BlkGemmPipeSched` placeholder that expands to the
    # fully-qualified enumerator. The rendered standalone kernel has no such local
    # alias, so the bare token would not resolve -- qualify it here.
    op_instances = [
        replace(
            op,
            is_wmma=True,
            block_gemm_pipeline_scheduler=(
                f"BlockGemmPipelineScheduler::{op.block_gemm_pipeline_scheduler}"
                if not str(op.block_gemm_pipeline_scheduler).startswith(
                    "BlockGemmPipelineScheduler::"
                )
                else op.block_gemm_pipeline_scheduler
            ),
        )
        for op in op_instances
    ]

    log.debug("ck batched WMMA instances from library: %d", len(op_instances))

    return canonical_instances(_substitute_scheduler_spec(op_instances))


if __name__ == "__main__":
    print(gen_ops_library())
