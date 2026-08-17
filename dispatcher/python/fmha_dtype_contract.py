#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional


class FmhaDTypeContractKind(Enum):
    HOMOGENEOUS = "homogeneous"
    ALL_FP8_WITH_BF16_OUTPUT = "all_fp8_with_bf16_output"
    ALL_FP8_WITH_FP32_OUTPUT = "all_fp8_with_fp32_output"
    MIXED_Q_FP8_KV = "mixed_q_fp8_kv"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class FmhaDTypeContract:
    data_type: str
    q_dtype: str
    k_dtype: str
    v_dtype: str
    o_dtype: str
    kind: FmhaDTypeContractKind

    @property
    def uses_fp8_kv(self) -> bool:
        return _is_fp8(self.k_dtype) and _is_fp8(self.v_dtype)


_TOKEN_CONTRACTS = {
    "fp16": ("fp16", "fp16", "fp16", "fp16"),
    "bf16": ("bf16", "bf16", "bf16", "bf16"),
    "fp32": ("fp32", "fp32", "fp32", "fp32"),
    "fp8": ("fp8", "fp8", "fp8", "fp8"),
    "bf8": ("bf8", "bf8", "bf8", "bf8"),
    "fp8bf16": ("fp8", "fp8", "fp8", "bf16"),
    "fp8fp32": ("fp8", "fp8", "fp8", "fp32"),
    "fp8fp16": ("fp8", "fp8", "fp8", "fp16"),
    "mxfp8": ("fp8", "fp8", "fp8", "fp32"),
}


def _normalize_dtype(dtype: Optional[str]) -> Optional[str]:
    if dtype is None:
        return None

    normalized = str(dtype).lower()
    aliases = {
        "float16": "fp16",
        "half": "fp16",
        "uint16": "bf16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "uint8": "fp8",
        "fp8_e4m3": "fp8",
        "fp8_e4m3fnuz": "fp8",
        "float8_e4m3fnuz": "fp8",
    }
    return aliases.get(normalized, normalized)


def _is_fp8(dtype: str) -> bool:
    return _normalize_dtype(dtype) in {"fp8", "bf8", "mxfp8"}


def _classify(
    q_dtype: str, k_dtype: str, v_dtype: str, o_dtype: str
) -> FmhaDTypeContractKind:
    q_dtype = _normalize_dtype(q_dtype) or ""
    k_dtype = _normalize_dtype(k_dtype) or ""
    v_dtype = _normalize_dtype(v_dtype) or ""
    o_dtype = _normalize_dtype(o_dtype) or ""

    if q_dtype == k_dtype == v_dtype == o_dtype:
        return FmhaDTypeContractKind.HOMOGENEOUS
    if _is_fp8(q_dtype) and _is_fp8(k_dtype) and _is_fp8(v_dtype):
        if o_dtype == "bf16":
            return FmhaDTypeContractKind.ALL_FP8_WITH_BF16_OUTPUT
        if o_dtype == "fp32":
            return FmhaDTypeContractKind.ALL_FP8_WITH_FP32_OUTPUT
    if (
        q_dtype in {"fp16", "bf16"}
        and _is_fp8(k_dtype)
        and _is_fp8(v_dtype)
        and o_dtype in {"fp16", "bf16"}
    ):
        return FmhaDTypeContractKind.MIXED_Q_FP8_KV
    return FmhaDTypeContractKind.UNSUPPORTED


def dtype_contract_from_components(
    data_type: str,
    q_dtype: str,
    k_dtype: str,
    v_dtype: str,
    o_dtype: str,
) -> FmhaDTypeContract:
    data_type = _normalize_dtype(data_type) or data_type
    q_dtype = _normalize_dtype(q_dtype) or ""
    k_dtype = _normalize_dtype(k_dtype) or ""
    v_dtype = _normalize_dtype(v_dtype) or ""
    o_dtype = _normalize_dtype(o_dtype) or ""
    return FmhaDTypeContract(
        data_type=data_type,
        q_dtype=q_dtype,
        k_dtype=k_dtype,
        v_dtype=v_dtype,
        o_dtype=o_dtype,
        kind=_classify(q_dtype, k_dtype, v_dtype, o_dtype),
    )


def dtype_contract_from_data_type(data_type: str) -> FmhaDTypeContract:
    data_type = _normalize_dtype(data_type) or data_type
    q_dtype, k_dtype, v_dtype, o_dtype = _TOKEN_CONTRACTS.get(
        data_type, (data_type, data_type, data_type, data_type)
    )
    return dtype_contract_from_components(data_type, q_dtype, k_dtype, v_dtype, o_dtype)


def dtype_contract_from_signature(signature: Mapping[str, object]) -> FmhaDTypeContract:
    data_type = str(signature.get("data_type", "fp16"))
    inferred = dtype_contract_from_data_type(data_type)
    kv_dtype = signature.get("kv_data_type", signature.get("kv_dtype"))

    q_dtype = signature.get("q_data_type", signature.get("q_dtype", inferred.q_dtype))
    k_dtype = signature.get(
        "k_data_type", signature.get("k_dtype", kv_dtype or inferred.k_dtype)
    )
    v_dtype = signature.get(
        "v_data_type", signature.get("v_dtype", kv_dtype or inferred.v_dtype)
    )
    o_dtype = signature.get("o_data_type", signature.get("o_dtype", inferred.o_dtype))

    return dtype_contract_from_components(
        data_type,
        str(q_dtype),
        str(k_dtype),
        str(v_dtype),
        str(o_dtype),
    )
