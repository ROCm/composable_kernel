// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>
#include <iostream>
#include <numeric>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <time.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/reduce.hpp"
#include "topk_softmax_decode_api.hpp"
#include "topk_softmax_api.hpp"
#include "moe_sorting_api.hpp"

#ifndef CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID
#define CK_TILE_REFERENCE_MOE_SORTING_MOCK_ID 1
#endif
#include "ck_tile/host/reference/reference_moe_sorting.hpp"

// CPU reference: softmax -> topk -> moe_sorting
template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
bool reference_fused(const ck_tile::HostTensor<InputType>& x_host,
                     ck_tile::index_t topk,
                     ck_tile::index_t num_experts,
                     ck_tile::index_t unit_size,
                     ck_tile::HostTensor<IndexType>& ref_sorted_ids,
                     ck_tile::HostTensor<WeightType>& ref_sorted_weights,
                     ck_tile::HostTensor<IndexType>& ref_sorted_expert_ids,
                     ck_tile::index_t& ref_unit_cnt)
{
    auto probs = ck_tile::reference_softmax<InputType, WeightType, WeightType>(x_host);

    ck_tile::HostTensor<WeightType> topk_vals({1, topk});
    ck_tile::HostTensor<IndexType> topk_idxs({1, topk});
    ck_tile::reference_topk(probs, topk_vals, topk_idxs, topk);

    ck_tile::HostTensor<IndexType> local_expert_mask({num_experts});
    ref_unit_cnt = 0;
    ck_tile::reference_moe_sorting<WeightType, IndexType>(
        topk_idxs,
        topk_vals,
        local_expert_mask,
        ref_sorted_ids,
        ref_sorted_weights,
        ref_sorted_expert_ids,
        ref_unit_cnt,
        num_experts,
        unit_size,
        1,
        false,
        true);

    return true;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "do CPU validation")
        .insert("pr_i", "bf16", "input data type: fp16/bf16")
        .insert("e", "128", "number of experts")
        .insert("k", "8", "topk")
        .insert("unit", "32", "unit_size (block_size_M)")
        .insert("model_dim", "7168", "model dimension for moe_buf zeroing")
        .insert("seed", "-1", "random seed, -1 = random")
        .insert("warmup", "5", "warmup iterations")
        .insert("repeat", "20", "benchmark iterations");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename InputType, typename WeightType, typename IndexType = ck_tile::index_t>
bool run_test(ck_tile::ArgParser args)
{
    int validate    = args.get_int("v");
    std::string pr  = args.get_str("pr_i");
    int experts     = args.get_int("e");
    int topk        = args.get_int("k");
    int unit_size   = args.get_int("unit");
    int model_dim   = args.get_int("model_dim");
    int seed        = args.get_int("seed");
    int warmup      = args.get_int("warmup");
    int repeat      = args.get_int("repeat");

    if(seed < 0)
        seed = std::time(nullptr);

    if(topk > experts)
    {
        printf("topk %d > experts %d, skip\n", topk, experts);
        return false;
    }

    int tokens                = 1;
    int max_num_tokens_padded = topk + experts * unit_size - topk;
    int max_num_m_blocks      = (max_num_tokens_padded + unit_size - 1) / unit_size;

    // Host tensors
    ck_tile::HostTensor<InputType> x_host({1, experts});
    {
        auto rng = ck_tile::FillUniformDistribution_Unique<InputType>{
            -5.f, 5.f, static_cast<uint32_t>(seed)};
        ck_tile::HostTensor<InputType> row({experts});
        rng(row);
        std::copy(row.begin(), row.end(), x_host.begin());
    }

    // ---------- Device buffers (shared) ----------
    ck_tile::DeviceMem x_dev(x_host.get_element_space_size_in_bytes());
    x_dev.ToDevice(x_host.data());

    // ---------- Fused kernel buffers ----------
    ck_tile::DeviceMem fused_sorted_ids(max_num_tokens_padded * sizeof(IndexType));
    ck_tile::DeviceMem fused_sorted_weights(max_num_tokens_padded * sizeof(WeightType));
    ck_tile::DeviceMem fused_sorted_expert_ids(max_num_m_blocks * sizeof(IndexType));
    ck_tile::DeviceMem fused_num_valid(2 * sizeof(IndexType));
    ck_tile::DeviceMem fused_moe_buf(model_dim * sizeof(WeightType));
    {
        std::vector<float> ones(model_dim, 1.0f);
        fused_moe_buf.ToDevice(ones.data());
    }

    // ---------- Two-kernel baseline buffers ----------
    ck_tile::DeviceMem topk_w_dev(topk * sizeof(WeightType));
    ck_tile::DeviceMem topk_i_dev(topk * sizeof(IndexType));
    ck_tile::DeviceMem base_sorted_ids(max_num_tokens_padded * sizeof(IndexType));
    ck_tile::DeviceMem base_sorted_weights(max_num_tokens_padded * sizeof(WeightType));
    ck_tile::DeviceMem base_sorted_expert_ids(max_num_m_blocks * sizeof(IndexType));
    ck_tile::DeviceMem base_num_valid(2 * sizeof(IndexType));
    ck_tile::DeviceMem base_moe_buf(model_dim * sizeof(WeightType));
    int ws_size = moe_sorting_get_workspace_size(tokens, experts, topk, 0);
    ck_tile::DeviceMem base_ws(ws_size > 0 ? ws_size : 1);
    if(ws_size > 0)
        base_ws.SetZero();

    ck_tile::stream_config sc{nullptr, true, 0, warmup, repeat};

    // ====================== Fused kernel ======================
    topk_softmax_decode_trait fused_trait{pr, "fp32", experts, "softmax"};
    topk_softmax_decode_kargs fused_karg{
        x_dev.GetDeviceBuffer(),
        experts,
        topk,
        experts,
        true,
        fused_sorted_ids.GetDeviceBuffer(),
        fused_sorted_weights.GetDeviceBuffer(),
        fused_sorted_expert_ids.GetDeviceBuffer(),
        fused_num_valid.GetDeviceBuffer(),
        fused_moe_buf.GetDeviceBuffer(),
        unit_size,
        model_dim,
        static_cast<int>(sizeof(WeightType))};

    float ms_fused = topk_softmax_decode(fused_trait, fused_karg, sc);

    // ============= Two-kernel baseline: topk + moe_sorting =============
    topk_softmax_trait ts_trait{pr, "fp32", experts, "softmax"};
    topk_softmax_kargs ts_karg{
        x_dev.GetDeviceBuffer(),
        topk_w_dev.GetDeviceBuffer(),
        topk_i_dev.GetDeviceBuffer(),
        tokens,
        experts,
        topk,
        experts,
        topk};

    moe_sorting_trait ms_trait{"int32", "fp32", false, true, 0};
    moe_sorting_args ms_arg{
        topk_i_dev.GetDeviceBuffer(),
        topk_w_dev.GetDeviceBuffer(),
        nullptr,
        nullptr,
        base_sorted_ids.GetDeviceBuffer(),
        base_sorted_weights.GetDeviceBuffer(),
        base_sorted_expert_ids.GetDeviceBuffer(),
        base_num_valid.GetDeviceBuffer(),
        base_moe_buf.GetDeviceBuffer(),
        ws_size > 0 ? base_ws.GetDeviceBuffer() : nullptr,
        tokens,
        unit_size,
        experts,
        topk,
        model_dim,
        static_cast<int>(sizeof(WeightType))};

    // Time the two kernels together using launch_kernel with two lambdas
    auto sc_sub = ck_tile::stream_config{nullptr, false, 0, 0, 1};
    float ms_baseline = ck_tile::launch_kernel(
        sc,
        [&](const ck_tile::stream_config&) { topk_softmax(ts_trait, ts_karg, sc_sub); },
        [&](const ck_tile::stream_config&) { moe_sorting(ms_trait, ms_arg, sc_sub); });

    float speedup = (ms_baseline > 0 && ms_fused > 0) ? ms_baseline / ms_fused : 0;
    printf("[%s] E:%d, k:%d, unit:%d  |  fused:%.4fms  baseline(topk+sort):%.4fms  speedup:%.2fx",
           pr.c_str(),
           experts,
           topk,
           unit_size,
           ms_fused,
           ms_baseline,
           speedup);

    if(ms_fused < 0 || ms_baseline < 0)
    {
        printf(" (not supported)\n");
        return false;
    }

    // ====================== Validation ======================
    bool pass = true;
    if(validate)
    {
        ck_tile::HostTensor<IndexType> sorted_ids_host({max_num_tokens_padded});
        ck_tile::HostTensor<WeightType> sorted_weights_host({max_num_tokens_padded});
        ck_tile::HostTensor<IndexType> sorted_expert_ids_host({max_num_m_blocks});
        ck_tile::HostTensor<IndexType> num_valid_host({2});
        std::vector<float> moe_buf_host(model_dim);

        fused_sorted_ids.FromDevice(sorted_ids_host.data());
        fused_sorted_weights.FromDevice(sorted_weights_host.data());
        fused_sorted_expert_ids.FromDevice(sorted_expert_ids_host.data());
        fused_num_valid.FromDevice(num_valid_host.data());
        fused_moe_buf.FromDevice(moe_buf_host.data());

        ck_tile::HostTensor<IndexType> ref_sorted_ids({max_num_tokens_padded});
        ck_tile::HostTensor<WeightType> ref_sorted_weights({max_num_tokens_padded});
        ck_tile::HostTensor<IndexType> ref_sorted_expert_ids({max_num_m_blocks});
        IndexType sentinel = static_cast<uint32_t>((1 & 0x00ffffff) | ((topk & 0xff) << 24));
        std::fill(ref_sorted_ids.begin(), ref_sorted_ids.end(), sentinel);
        std::fill(ref_sorted_weights.begin(), ref_sorted_weights.end(), WeightType(0));
        std::fill(ref_sorted_expert_ids.begin(), ref_sorted_expert_ids.end(), -1);

        ck_tile::index_t ref_unit_cnt = 0;
        reference_fused<InputType, WeightType, IndexType>(
            x_host, topk, experts, unit_size,
            ref_sorted_ids, ref_sorted_weights, ref_sorted_expert_ids, ref_unit_cnt);

        int num_valid_padded = num_valid_host(0);
        int num_valid_tokens = num_valid_host(1);

        if(num_valid_padded != ref_unit_cnt)
        {
            printf(" FAIL:num_valid[0] got %d ref %d;", num_valid_padded, ref_unit_cnt);
            pass = false;
        }
        if(num_valid_tokens != 1)
        {
            printf(" FAIL:num_valid[1] got %d;", num_valid_tokens);
            pass = false;
        }

        int n_tiles = num_valid_padded / unit_size;
        for(int i = 1; i < n_tiles; i++)
        {
            if(sorted_expert_ids_host(i) < sorted_expert_ids_host(i - 1))
            {
                printf(" FAIL:expert_ids not ascending;");
                pass = false;
                break;
            }
        }

        WeightType wsum = 0;
        for(int i = 0; i < topk; i++)
            wsum += sorted_weights_host(i * unit_size);
        if(std::abs(wsum - 1.0f) > 1e-3f)
        {
            printf(" FAIL:wsum=%.6f;", static_cast<float>(wsum));
            pass = false;
        }

        bool buf_zeroed = std::all_of(
            moe_buf_host.begin(), moe_buf_host.end(), [](float v) { return v == 0.0f; });
        if(!buf_zeroed)
        {
            printf(" FAIL:moe_buf not zeroed;");
            pass = false;
        }
    }

    printf("  valid:%s\n", pass ? "y" : "n");
    fflush(stdout);
    return pass;
}

int main(int argc, char** argv)
{
    auto [result, args] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string pr = args.get_str("pr_i");
    bool r         = true;

    if(pr == "fp16")
        r &= run_test<ck_tile::fp16_t, float>(args);
    else if(pr == "bf16")
        r &= run_test<ck_tile::bf16_t, float>(args);
    else
    {
        printf("unsupported pr_i: %s\n", pr.c_str());
        return -1;
    }

    return r ? 0 : -1;
}
