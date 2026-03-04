// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"

template <typename T>
using GemmConfig = GemmConfigQuantPrefill<T>;

#define RUN_GEMM_EXAMPLE_PREC_TYPE                            \
    run_gemm_example_prec_type<GemmConfig<ck_tile::pk_fp4_t>, \
                               TypeConfig,                    \
                               QuantGroupSize,                \
                               ck_tile::QuantType::BQuantGrouped>(arg_parser);

static auto _ = []() {
    auto& lut        = get_kernel_lut();
    using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::bf16_t,
                                                    ck_tile::pk_fp4_t,
                                                    ck_tile::bf16_t,
                                                    ck_tile::e8m0_t>{});

    lut[hash_multiple_strings(
        {"mxbf16fp4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x1x32"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 32>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings(
        {"mxbf16fp4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x1x64"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings(
        {"mxbf16fp4", "bquant", "non-preshuffleb", "non-preshufflequant", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    return 0;
}();
