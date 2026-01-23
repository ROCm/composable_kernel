// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "run_gemm_quant_example.inc"

template <typename T>
using GemmConfig = GemmConfigABQuantPrefill<T>;

template <typename T>
using GemmConfigPreshuffleB = GemmConfigPreshuffleB_ABQuant_Prefill<T>;

// template <typename T>
// using GemmConfigPreshuffleB = GemmConfigPreshuffleB_ABQuant_Decode<T>;

void abquant_quantgrouped_fp4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void abquant_quantgrouped_fp8_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void abquant_quantgrouped_bf8_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void abquant_quantgrouped_preshuffleb_fp4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);

void abquant_quantgrouped_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
{
    abquant_quantgrouped_fp4_instance_factory(lut);
    abquant_quantgrouped_fp8_instance_factory(lut);
    abquant_quantgrouped_bf8_instance_factory(lut);
    abquant_quantgrouped_preshuffleb_fp4_instance_factory(lut);
}
