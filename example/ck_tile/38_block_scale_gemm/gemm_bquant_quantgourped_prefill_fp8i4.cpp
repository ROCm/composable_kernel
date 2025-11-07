// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// This example demonstrates 2D block scale quantization (N×K) for BQuant
// using non-preshuffled configuration.
// NOTE: Once more 2d support is ready, we can migrate all 2d quant types to this example
// This is currently done separately to avoid too verbose dispatching.

#include "run_gemm_quant_example.inc"

template <typename T>
using GemmConfig = GemmConfigBQuantPrefill<T>;

#define RUN_GEMM_EXAMPLE_PREC_TYPE                         \
    run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>, \
                               TypeConfig,                 \
                               QuantGroupSize,             \
                               ck_tile::QuantType::BQuantGrouped>(arg_parser);

void bquant_quantgrouped_fp8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(ck_tile::ArgParser&)>>& lut)
{
    using TypeConfig = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
                                                    ck_tile::pk_int4_t,
                                                    ck_tile::half_t,
                                                    ck_tile::fp8_t>{});
    lut[hash_multiple_strings({"fp8i4", "bquant", "non-preshuffleb", "1x1x64"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 64>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings({"fp8i4", "bquant", "non-preshuffleb", "1x1x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings({"fp8i4", "bquant", "non-preshuffleb", "1x8x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 8, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings({"fp8i4", "bquant", "non-preshuffleb", "1x32x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 32, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
    lut[hash_multiple_strings({"fp8i4", "bquant", "non-preshuffleb", "1x64x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using QuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 64, 128>>;
            return RUN_GEMM_EXAMPLE_PREC_TYPE;
        };
}
