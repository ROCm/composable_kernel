// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"
#include "gemm_abquant_quantgrouped.h"

static auto _ = []() {
    auto& lut = get_kernel_lut();

    lut[hash_multiple_strings({"bf8",
                               "abquant",
                               "preshuffleb",
                               "non-preshufflequant",
                               "1x1x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfigPrefill<ck_tile::bf8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
    lut[hash_multiple_strings({"bf8",
                               "abquant",
                               "preshuffleb",
                               "non-preshufflequant",
                               "1x128x128"})] = [](const ck_tile::ArgParser& arg_parser) {
        using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
        using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
        using TypeConfig =
            decltype(GemmQuantTypeConfig<ck_tile::bf8_t, ck_tile::bf8_t, ck_tile::half_t, float>{});
        return run_gemm_example_prec_type<GemmConfigPrefill<ck_tile::bf8_t>,
                                          TypeConfig,
                                          AQuantGroupSize,
                                          BQuantGroupSize,
                                          ck_tile::QuantType::ABQuantGrouped>(arg_parser);
    };
    return 0;
}();
