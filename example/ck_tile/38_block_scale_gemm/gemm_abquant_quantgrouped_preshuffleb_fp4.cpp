// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "38_block_scale_gemm/gemm_utils.hpp"
#include "run_gemm_quant_example.inc"

template <typename T>
using GemmConfig = GemmConfigABQuantPrefill<T>;

template <typename T>
using GemmConfigPreshuffleB = GemmConfigPreshuffleB_ABQuant_Prefill<T>;

void abquant_quantgrouped_preshuffleb_fp4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
{
    lut[hash_multiple_strings(
        {"fp4", "abquant", "preshuffleb", "non-preshufflequant", "1x128x128"})] =
        [](const ck_tile::ArgParser& arg_parser) {
            using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
            using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 128, 128>>;
            using TypeConfig      = decltype(GemmQuantTypeConfig<ck_tile::pk_fp4_t,
                                                                 ck_tile::pk_fp4_t,
                                                                 ck_tile::half_t,
                                                                 float>{});
            return run_gemm_example_prec_type<GemmConfigPreshuffleB<ck_tile::pk_fp4_raw_t>,
                                              TypeConfig,
                                              AQuantGroupSize,
                                              BQuantGroupSize,
                                              ck_tile::QuantType::ABQuantGrouped>(arg_parser);
        };
}
