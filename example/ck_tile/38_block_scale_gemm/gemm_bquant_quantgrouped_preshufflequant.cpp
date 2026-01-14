// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"

template <typename T>
using GemmConfig = GemmConfigPreshuffleBQuantPrefill<T>;

void bquant_quantgrouped_preshufflequant_fp8_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshufflequant_fp8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshufflequant_bf8_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshufflequant_bf8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);

void bquant_quantgrouped_preshufflequant_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
{
    bquant_quantgrouped_preshufflequant_fp8_instance_factory(lut);
    bquant_quantgrouped_preshufflequant_bf8_instance_factory(lut);
    bquant_quantgrouped_preshufflequant_fp8i4_instance_factory(lut);
    bquant_quantgrouped_preshufflequant_bf8i4_instance_factory(lut);
}
