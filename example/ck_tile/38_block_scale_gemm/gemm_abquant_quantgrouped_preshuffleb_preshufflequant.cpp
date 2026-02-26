// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "38_block_scale_gemm/gemm_utils.hpp"
#include "run_gemm_quant_example.inc"

template <typename T, bool TransposeC>
using GemmConfigPreshuffleB_PreshuffleBQuant =
    GemmConfigPreshuffleB_ABQuant_PreshuffleBQuant_Prefill<T, TransposeC>;

static auto _ = []() {
    auto& lut = get_kernel_lut();
    lut[hash_multiple_strings({"fp8", "abquant", "preshuffleb", "preshufflequant", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
            using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
            using TypeConfig      = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
                                                                 ck_tile::fp8_t,
                                                                 ck_tile::half_t,
                                                                 float>{});
            return run_gemm_example_prec_type<
                GemmConfigPreshuffleB_PreshuffleBQuant<ck_tile::fp8_t, false>,
                TypeConfig,
                AQuantGroupSize,
                BQuantGroupSize,
                ck_tile::QuantType::ABQuantGrouped>(arg_parser);
        };
    lut[hash_multiple_strings({"fp8", "abquant", "preshuffleb", "preshufflequant", "1x128x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
            using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
            using TypeConfig      = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
                                                                 ck_tile::fp8_t,
                                                                 ck_tile::half_t,
                                                                 float>{});
            return run_gemm_example_prec_type<
                GemmConfigPreshuffleB_PreshuffleBQuant<ck_tile::fp8_t, true>,
                TypeConfig,
                AQuantGroupSize,
                BQuantGroupSize,
                ck_tile::QuantType::ABQuantGrouped>(arg_parser);
        };
    return 0;
}();
