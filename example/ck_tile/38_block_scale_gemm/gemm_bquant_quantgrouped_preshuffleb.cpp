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

void bquant_quantgrouped_preshuffleb_fp8_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshuffleb_fp8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshuffleb_bf8_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
void bquant_quantgrouped_preshuffleb_bf8i4_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);

void bquant_quantgrouped_preshuffleb_instance_factory(
    std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
{
    bquant_quantgrouped_preshuffleb_fp8_instance_factory(lut);
    bquant_quantgrouped_preshuffleb_fp8i4_instance_factory(lut);
    bquant_quantgrouped_preshuffleb_bf8_instance_factory(lut);
    bquant_quantgrouped_preshuffleb_bf8i4_instance_factory(lut);
}
