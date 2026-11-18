// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"

template <typename T>
using GemmConfig = GemmConfigQuantDecode<T>;

template <typename Fp8, typename QuantGroupSize>
int run_fp8_bquant(const ck_tile::ArgParser& arg_parser)
{
    using TypeConfig = GemmQuantTypeConfig<Fp8, Fp8, ck_tile::half_t, float>;
    return run_gemm_example_prec_type<GemmConfig<Fp8>,
                                      TypeConfig,
                                      QuantGroupSize,
                                      ck_tile::QuantType::BQuantGrouped>(arg_parser);
}

template <typename QuantGroupSize>
int run_native_fp8_bquant(const ck_tile::ArgParser& arg_parser)
{
#if CK_TILE_USE_OCP_FP8
    if(ck_tile::get_device_name() == "gfx942")
        return run_fp8_bquant<ck_tile::fp8_fnuz_t, QuantGroupSize>(arg_parser);
#endif
    return run_fp8_bquant<ck_tile::fp8_t, QuantGroupSize>(arg_parser);
}

static auto _ = []() {
    auto& lut = get_kernel_lut();
#ifndef CK_GFX950_SUPPORT
    lut[hash_multiple_strings(
        {"fp8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x1x64"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;
            return run_native_fp8_bquant<QuantGroupSize>(arg_parser);
        };
#endif
    lut[hash_multiple_strings(
        {"fp8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
            return run_native_fp8_bquant<QuantGroupSize>(arg_parser);
        };
    lut[hash_multiple_strings(
        {"fp8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x8x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 8, 128>>;
            return run_native_fp8_bquant<QuantGroupSize>(arg_parser);
        };
    lut[hash_multiple_strings(
        {"fp8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x32x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 32, 128>>;
            return run_native_fp8_bquant<QuantGroupSize>(arg_parser);
        };
    lut[hash_multiple_strings(
        {"fp8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x64x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;
            return run_native_fp8_bquant<QuantGroupSize>(arg_parser);
        };
    lut[hash_multiple_strings(
        {"fp8", "bquant", "non-preshuffleb", "non-preshufflequant", "1x128x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
            return run_native_fp8_bquant<QuantGroupSize>(arg_parser);
        };
    return 0;
}();
