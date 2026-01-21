// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"

#if CK_TILE_USE_WMMA
template <typename T>
using GemmConfig = GemmConfigPreshuffleB_BQuant_Prefill_Wmma<T>;
#else
template <typename T>
using GemmConfig = GemmConfigPreshuffleB_BQuant_Prefill<T>;
#endif

#define RUN_GEMM_EXAMPLE_PREC_TYPE                         \
    run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>, \
                               TypeConfig,                 \
                               QuantGroupSize,             \
                               ck_tile::QuantType::BQuantGrouped>(arg_parser);

static auto _ = []() {
    auto& lut = get_kernel_lut();
    using TypeConfig =
        decltype(GemmQuantTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t, float>{});
    lut[hash_multiple_strings({"fp8", "bquant", "preshuffleb", "non-preshufflequant", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings({"fp8", "bquant", "preshuffleb", "non-preshufflequant", "1x8x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 8, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings(
        {"fp8", "bquant", "preshuffleb", "non-preshufflequant", "1x32x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 32, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings(
        {"fp8", "bquant", "preshuffleb", "non-preshufflequant", "1x64x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings(
        {"fp8", "bquant", "preshuffleb", "non-preshufflequant", "1x128x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    return 0;
}();
