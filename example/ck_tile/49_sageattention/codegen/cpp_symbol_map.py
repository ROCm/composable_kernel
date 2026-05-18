# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# generate kernel instances to speed up compilation
FWD_DTYPE_MAP = {
    "fp16": "SageAttentionFwdFp16",
    "bf16": "SageAttentionFwdBf16",
    "fp8bf16": "SageAttentionFwdFp8Bf16",
    "i8fp8bf16": "SageAttentionFwdI8Fp8Bf16",
    "i4fp8bf16": "SageAttentionFwdI4Fp8Bf16",
}

_MASK_SIMPLIFIED_MAP = {
    "s_no": "ck_tile::SimplifiedGenericAttentionMask<false>",
    "s_mask": "ck_tile::SimplifiedGenericAttentionMask<true>",
}

_MASK_MAP = {
    "no": "SageAttnMasks::NoMask",
    "causal": "SageAttnMasks::CausalMask",
    "generic": "SageAttnMasks::GenericMask",
}


def get_mask_map(mask_impl: str):
    if mask_impl == "generic":
        return _MASK_MAP
    elif mask_impl == "simplified":
        return _MASK_SIMPLIFIED_MAP
    else:
        assert False
        return None


def get_mask_impl(mask: str) -> str:
    return "simplified" if mask.startswith("s_") else "generic"


def get_mask_cpp_type(mask: str) -> str:
    return get_mask_map(get_mask_impl(mask))[mask]


_MASK_CHECK_MAP = {
    "no": "t.mask_type == mask_enum::no_mask",
    "causal": "t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right",
    "generic": "t.mask_type == mask_enum::window_generic",
}

_MASK_SIMPLIFIED_CHECK_MAP = {
    "s_no": "t.mask_type == mask_enum::no_mask",
    "s_mask": "t.mask_type != mask_enum::no_mask",
}


def get_mask_check_map(mask: str):
    if mask == "generic":
        return _MASK_CHECK_MAP
    elif mask == "simplified":
        return _MASK_SIMPLIFIED_CHECK_MAP
    else:
        assert False
        return None


def get_mask_cpp_check_expr(mask: str) -> str:
    return get_mask_check_map(get_mask_impl(mask))[mask]


QSCALE_MAP = {
    "no": "ck_tile::BlockSageAttentionQuantScaleEnum::NO_SCALE",
    "pertensor": "ck_tile::BlockSageAttentionQuantScaleEnum::PERTENSOR",
    "blockscale": "ck_tile::BlockSageAttentionQuantScaleEnum::BLOCKSCALE",
    "perwarp": "ck_tile::BlockSageAttentionQuantScaleEnum::PERWARP",
    "perthread": "ck_tile::BlockSageAttentionQuantScaleEnum::PERTHREAD",
}

QSCALE_CHECK_MAP = {
    "no": "quant_scale_enum::no_scale",
    "pertensor": "quant_scale_enum::pertensor",
    "blockscale": "quant_scale_enum::blockscale",
    "perwarp": "quant_scale_enum::perwarp",
    "perthread": "quant_scale_enum::perthread",
}

MODE_MAP = {"batch": "false", "group": "true"}

LAYOUT_MAP = {"row": "true", "col": "false"}

PIPELINE_MAP = {
    "qr": "ck_tile::BlockSageAttentionPipelineQRKSVS",
    "qr_async": "ck_tile::BlockSageAttentionPipelineQRKSVSAsync",
}

PIPELINE_ENUM_MAP = {
    "qr": "ck_tile::BlockSageAttnPipelineEnum::QRKSVS",
    "qr_async": "ck_tile::BlockSageAttnPipelineEnum::QRKSVS_ASYNC",
}

BOOL_MAP = {
    "t": "true",
    "f": "false",
    True: "true",
    False: "false",
}
