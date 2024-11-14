// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

// A async load to LDS, B direct to AGPR
// B matrix preshuffled in br*kr*w
// require 4 wave, occupancy=1c
// agpr useage:256
// vgpr usage:64(A local) + 64(acc) + 8(os_a) + 8(os_b) = 144 (rem:112)
//
// for this gemm, 4 16x16x16 transposed layout
//  input A vpgpr layout
//   v0-v15: [ 0:15](gemm_m)x128(gemm_k)
//  v16-v31: [16:31](gemm_m)x128(gemm_k)

//  input B vpgpr layout
//   v0-v15: [  0: 15](gemm_n)x128(gemm_k)
//  v16-v31: [ 64: 79](gemm_n)x128(gemm_k)
//  ......................
//  v111-v127: [448:463](gemm_n)x128(gemm_k)

//  output C vpgpr layout
//   v0-v3 : [ 0:15](gemm_m)x[ 0: 15](gemm_n)
//   v4-v7 : [16:31](gemm_m)x[ 0: 15](gemm_n)
//   v8-v11: [ 0:15](gemm_m)x[64: 79](gemm_n)
//  v12-v15: [16:31](gemm_m)x[64: 79](gemm_n)
//  ......................
//  v56-v59: [ 0:15](gemm_m)x[448:463](gemm_n)
//  v60-v63: [16:31](gemm_m)x[448:463](gemm_n)
struct FlatmmUK_GFX9_32x512x128_1x4x1_16x16x16_BF16
{
    static constexpr index_t Block_M = 32;
    static constexpr index_t Block_N = 512;
    static constexpr index_t Block_K = 128;

    static constexpr index_t WarpPerBlock_M = 1;
    static constexpr index_t WarpPerBlock_N = 4;
    static constexpr index_t WarpPerBlock_K = 1;

    static constexpr index_t NumWarps = 4;

    static constexpr index_t Warp_M = 16;
    static constexpr index_t Warp_N = 16;
    static constexpr index_t Warp_K = 32; // 16 * SubKPacks

    static constexpr index_t BlockSize = 256;

    static constexpr index_t SubKPacks = 2; // this is used to gurantee every threads can do dwordx4

    // TODO: note Nr/Kr/W need consider SubKPacks
    static constexpr index_t Block_W  = Warp_N * Warp_K;  // 512 element
    static constexpr index_t Block_Nr = Block_N / Warp_N; // 32 element, 4 per wave
    static constexpr index_t Block_Kr = Block_K / Warp_K; // 4

    static constexpr index_t Repeat_M = Block_M / (Warp_M * WarpPerBlock_M); // 2
    static constexpr index_t Repeat_N = Block_N / (Warp_N * WarpPerBlock_N); // 8
    static constexpr index_t Repeat_K = Block_K / (Warp_K * WarpPerBlock_K); // 8/2=4

    static CK_TILE_DEVICE constexpr auto MakeCBlockDist()
    {
        constexpr auto c_block_outer_dstr_encoding = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<Repeat_M, WarpPerBlock_M>, sequence<Repeat_N, WarpPerBlock_N>>,
            tuple<sequence<1, 2>>,
            tuple<sequence<1, 1>>,
            sequence<2, 1>, // !! note here is different
            sequence<0, 0>>{};

        using WG        = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution;

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        return c_block_dstr;
    }

    static CK_TILE_DEVICE constexpr auto MakeCBlockTile()
    {
        using CDataType = float;
        constexpr auto c_block_dstr = MakeCBlockDist();
        auto c_block_tensor         = make_static_distributed_tensor<CDataType>(c_block_dstr);
        return c_block_tensor;
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsStoreDesc_A()
    {
        // A async->LDS
        // constexpr index_t Block_M = Problem::BlockShape::Block_M0;
        // constexpr index_t Block_K = Problem::BlockShape::Block_K0;
        // constexpr index_t BlockSize = Problem::BlockShape::BlockSize;
        constexpr index_t warpSize = ck_tile::get_warp_size();
        // constexpr index_t NumWarps = Problem::BlockShape::NumWarps;

        constexpr index_t KPack_  = 8;      // GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t KVector = 2;      // GetAlignment_A<Problem>(); // async copy 1 dword
        constexpr index_t KPad    = KPack_; // pad between warps

        static_assert(Block_K % KVector == 0);
        constexpr index_t LanesPerK = Block_K / KVector; // how many thread loading K
        if constexpr(LanesPerK >= warpSize)
        {
            // need multiple waves to load K
            static_assert(LanesPerK % warpSize == 0);
            constexpr index_t wavesPerK = LanesPerK / warpSize;
            if constexpr(wavesPerK > NumWarps)
            {
                // TODO: need multiple issues along K to load all data
            }
            else
            {
                constexpr index_t wavesPerM     = NumWarps / wavesPerK;
                constexpr index_t NumIssues     = Block_M / wavesPerM;
                constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<NumIssues>{},                             // m0
                               number<wavesPerM>{},                             // m1
                               number<wavesPerK>{},                             // k0
                               number<warpSize>{},                              // k1
                               number<KVector>{}),                              // k2
                    make_tuple(number<NumWarps*(warpSize * KVector + KPad)>{},  // m0
                               number<wavesPerK*(warpSize * KVector + KPad)>{}, // m1
                               number<warpSize * KVector + KPad>{},             // k0
                               number<KVector>{},                               // k1
                               number<1>{}),                                    // k2
                    number<KVector>{}, // lds store vector(actually no explicit store)
                    number<1>{});

                constexpr auto lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
                    lds_block_desc_0,
                    make_tuple(
                        make_pass_through_transform(number<NumIssues>{}),
                        make_merge_transform(make_tuple(number<wavesPerM>{}, number<wavesPerK>{})),
                        make_merge_transform(make_tuple(number<warpSize>{}, number<KVector>{}))),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                return lds_block_desc_issues_warps_lanes;
            }
        }
        else
        {
            // lanes within a wave load different M but same K
            static_assert(warpSize % LanesPerK == 0);
            constexpr index_t LaneGroups = warpSize / LanesPerK; // along m
            constexpr index_t NumIssues  = Block_M / (LaneGroups * NumWarps);

            constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                make_tuple(number<NumIssues>{},                            // m0
                           number<LaneGroups>{},                           // m1
                           number<NumWarps>{},                             // m2
                           number<LanesPerK>{},                            // k0
                           number<KVector>{}),                             // k1
                make_tuple(number<NumWarps*(warpSize * KVector + KPad)>{}, // m0
                           number<Block_K>{},                              // m1
                           number<warpSize * KVector + KPad>{},            // m2
                           number<KVector>{},                              // k0
                           number<1>{}),                                   // k1
                number<KVector>{}, // lds store vector(actually no explicit store)
                number<1>{});

            constexpr auto lds_block_desc_issues_warps_lanes = transform_tensor_descriptor(
                lds_block_desc_0,
                make_tuple(make_pass_through_transform(number<NumIssues>{}),
                           make_pass_through_transform(number<NumWarps>{}),
                           make_merge_transform(make_tuple(
                               number<LaneGroups>{}, number<LanesPerK>{}, number<KVector>{}))),
                make_tuple(sequence<0>{}, sequence<2>{}, sequence<1, 3, 4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

            return lds_block_desc_issues_warps_lanes;
        }
    }

    // template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsLoadDesc_A()
    {
        // load from LDS to register, every wave has same layout
        constexpr index_t KPack_  = 8;      // GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t KPad    = KPack_; // pad between warps

        constexpr index_t kAMLane     = 16;
        constexpr index_t kABKLane    = 4;
        constexpr index_t kABKPerLane = 4;
        constexpr index_t kKIter      = 2;
        static_assert(KPack_ == (kABKPerLane * kKIter));

        constexpr auto lds_block_desc_0 = make_naive_tensor_descriptor(
                    make_tuple(number<Repeat_M>{},            // m0 y
                               number<kAMLane>{},             // m1 p
                               number<Repeat_K>{},            // k0 y
                               number<kABKLane>{},            // k1 p
                               number<KPack_>{}),             // k2 y-vector
                    make_tuple(number<kAMLane*(Block_K + KPad)>{}, // m0
                               number<Block_K + KPad>{},           // m1
                               number<kABKLane * KPack_>{},        // k0
                               number<KPack_>{},                   // k1
                               number<1>{}),                       // k2
                    number<KPack_>{},                              // lds load vector
                    number<1>{});

        constexpr auto lds_desc_m_k = transform_tensor_descriptor(
            lds_block_desc_0,
            make_tuple(make_merge_transform(
                            make_tuple(number<Repeat_M>{}, number<kAMLane>{})),
                        make_merge_transform(make_tuple(
                            number<Repeat_K>{}, number<kABKLane>{}, number<KPack_>{}))),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc_m_k;
    }

    static constexpr auto GetGemm_AWarpEnc()
    {
        constexpr index_t kAMLane     = 16;
        constexpr index_t kABKLane    = 4;
        constexpr index_t kABKPerLane = 4;
        constexpr index_t kKIter      = 2;

        using enc_ = tile_distribution_encoding<
            sequence<>,
            tuple<sequence<kAMLane>, sequence<kABKLane, kABKPerLane * kKIter>>,
            tuple<sequence<2, 1>>,
            tuple<sequence<0, 0>>,
            sequence<2>,
            sequence<1>>;
        return enc_{};
    }

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return 32 * (128 + 8) * sizeof(bf16_t);
    }

    // TODO: need paired with tile_window_linear!
    // TODO: need call init_raw() before call this function!
    template <typename ARes, typename ACoords, typename BRes, typename BCoords>
    CK_TILE_DEVICE auto
    operator()(const ARes& res_a,
               const ACoords& cached_coords_a,
               const BRes& res_b,
               const BCoords& cached_coords_b,
               CK_TILE_LDS_ADDR void* smem,
               index_t k,
               index_t tile_offset_a, // for each tile, the offset to move for each unroll
               index_t tile_offset_b) // for each tile, the offset to move for each unroll
    {
        static_assert(ACoords::size() == Block_M * Block_K / BlockSize / 2 /*2x per dword*/); // 8
        static_assert(BCoords::size() == Repeat_N);

        auto a_sst = make_tile_window(
            make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<CK_TILE_LDS_ADDR bf16_t*>(smem), MakeLdsStoreDesc_A()),
            MakeLdsStoreDesc_A().get_lengths(),
            {0, 0, 0});

        auto a_sld = [&]() {
            constexpr auto a_warp_enc_      = GetGemm_AWarpEnc();
            constexpr auto a_outer_dstr_enc = tile_distribution_encoding<
                sequence<WarpPerBlock_N>,
                tuple<sequence<Repeat_M, WarpPerBlock_M>, sequence<Repeat_K>>,
                tuple<sequence<1, 0>>,
                tuple<sequence<1, 0>>,
                sequence<1, 2>,
                sequence<0, 0>>{};
            constexpr auto a_block_dstr_encode =
                detail::make_embed_tile_distribution_encoding(a_outer_dstr_enc, a_warp_enc_);
            return make_tile_window_linear(
                make_tensor_view<address_space_enum::lds>(
                    reinterpret_cast<CK_TILE_LDS_ADDR bf16_t*>(smem), MakeLdsLoadDesc_A()),
                MakeLdsLoadDesc_A().get_lengths(),
                {0, 0},
                make_static_tile_distribution(a_block_dstr_encode));
        }();

        const index_t tile_offset_a_bytes = tile_offset_a * sizeof(bf16_t);
        const index_t tile_offset_b_bytes = tile_offset_b * sizeof(bf16_t);

        const auto [m0_init_value, size_per_issue] = get_async_store_smem_info(a_sst);
        constexpr auto smem_buf_size =
            MakeLdsLoadDesc_A().get_element_space_size() * sizeof(bf16_t);
        static_assert(a_sld.get_num_of_access() == 8);
        constexpr auto sld_os = generate_tuple(
            [&](auto i_access) {
                return number<a_sld.get_bottom_linear_offset(i_access) * sizeof(bf16_t)>{};
            },
            number<a_sld.get_num_of_access()>{});


        // printf("----- tid:%d, a_sld:%d\n", static_cast<index_t>(threadIdx.x),
        //                 static_cast<index_t>(a_sld.cached_coords_[number<0>{}].get_offset()));



        index_t loop_cnt = k / Block_K;

        // this is the acc thread buffer
        fp32x4_t v_acc[16]{.0f};

        // B nr->kr
        // clang-format off
        _Pragma("clang diagnostic push");
        _Pragma("clang diagnostic ignored \"-Winline-asm\"");
        asm volatile(
            "s_mov_b32 s16,    %[s_res_a0] \n"
            "s_mov_b32 s17,    %[s_res_a1] \n"
            "s_mov_b32 s18,    %[s_res_a2] \n"
            "s_mov_b32 s19,    %[s_res_a3] \n"
            "s_mov_b32 s20,    %[s_res_b0] \n"
            "s_mov_b32 s21,    %[s_res_b1] \n"
            "s_mov_b32 s22,    %[s_res_b2] \n"
            "s_mov_b32 s23,    %[s_res_b3] \n"
            // "s_nop  4\n"
            "; -- prefetch A0\n"
            "s_add_u32     m0, 0, %[s_m0_init]                        \n"
            "buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                \n"
            "buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                \n"
            "buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                \n"
            "buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                \n"
            "buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                \n"
            "buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                \n"
            "buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                \n"
            "buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds    \n"
            "s_add_u32 m0, %[smem_sz], %[s_m0_init]                       \n"
            "s_cmp_gt_i32  %[s_loop_cnt] 1             ; move a with cond \n"
            "s_cselect_b32 s86, %[s_tile_os_a], 0      ; move a with cond  \n"
            "s_add_u32     s16, s86, s16               ; move a with cond \n"
            "s_addc_u32    s17, 0, s17                 ; move a with cond \n"
            "; -- prefetch A1\n"
            "buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds    \n"
            "s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds    \n"
            "s_add_u32 m0, 0, %[s_m0_init]                                \n"
            "s_cmp_gt_i32  %[s_loop_cnt] 2             ; move a with cond \n"
            "s_cselect_b32 s86, %[s_tile_os_a], 0      ; move a with cond  \n"
            "s_add_u32     s16, s86, s16               ; move a with cond \n"
            "s_addc_u32    s17, 0, s17                 ; move a with cond \n"
            "; -- prefetch B0\n"
            "buffer_load_dwordx4  acc[0:3], %[v_os_b0], s[20:23], 0 offen \n"
            "buffer_load_dwordx4  acc[4:7], %[v_os_b0], s[20:23], 0 offen offset:1024  \n"
            "buffer_load_dwordx4  acc[8:11], %[v_os_b0], s[20:23], 0 offen offset:2048  \n"
            "buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[20:23], 0 offen offset:3072  \n"
            "buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[20:23], 0 offen  \n"
            "buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[20:23], 0 offen offset:1024  \n"
            "buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[20:23], 0 offen offset:2048  \n"
            "buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[20:23], 0 offen offset:3072  \n"
            "buffer_load_dwordx4  acc[32:35], %[v_os_b2], s[20:23], 0 offen  \n"
            "buffer_load_dwordx4  acc[36:39], %[v_os_b2], s[20:23], 0 offen offset:1024  \n"
            "buffer_load_dwordx4  acc[40:43], %[v_os_b2], s[20:23], 0 offen offset:2048  \n"
            "buffer_load_dwordx4  acc[44:47], %[v_os_b2], s[20:23], 0 offen offset:3072  \n"
            "buffer_load_dwordx4  acc[48:51], %[v_os_b3], s[20:23], 0 offen  \n"
            "buffer_load_dwordx4  acc[52:55], %[v_os_b3], s[20:23], 0 offen offset:1024    \n"
            "buffer_load_dwordx4  acc[56:59], %[v_os_b3], s[20:23], 0 offen offset:2048    \n"
            "buffer_load_dwordx4  acc[60:63], %[v_os_b3], s[20:23], 0 offen offset:3072    \n"
            "buffer_load_dwordx4  acc[64:67], %[v_os_b4], s[20:23], 0 offen                \n"
            "buffer_load_dwordx4  acc[68:71], %[v_os_b4], s[20:23], 0 offen offset:1024    \n"
            "buffer_load_dwordx4  acc[72:75], %[v_os_b4], s[20:23], 0 offen offset:2048    \n"
            "buffer_load_dwordx4  acc[76:79], %[v_os_b4], s[20:23], 0 offen offset:3072    \n"
            "buffer_load_dwordx4  acc[80:83], %[v_os_b5], s[20:23], 0 offen                \n"
            "buffer_load_dwordx4  acc[84:87], %[v_os_b5], s[20:23], 0 offen offset:1024    \n"
            "buffer_load_dwordx4  acc[88:91], %[v_os_b5], s[20:23], 0 offen offset:2048    \n"
            "buffer_load_dwordx4  acc[92:95], %[v_os_b5], s[20:23], 0 offen offset:3072    \n"
            "buffer_load_dwordx4  acc[96:99], %[v_os_b6], s[20:23], 0 offen                \n"
            "buffer_load_dwordx4  acc[100:103], %[v_os_b6], s[20:23], 0 offen offset:1024  \n"
            "buffer_load_dwordx4  acc[104:107], %[v_os_b6], s[20:23], 0 offen offset:2048  \n"
            "buffer_load_dwordx4  acc[108:111], %[v_os_b6], s[20:23], 0 offen offset:3072  \n"
            "buffer_load_dwordx4  acc[112:115], %[v_os_b7], s[20:23], 0 offen              \n"
            "buffer_load_dwordx4  acc[116:119], %[v_os_b7], s[20:23], 0 offen offset:1024  \n"
            "buffer_load_dwordx4  acc[120:123], %[v_os_b7], s[20:23], 0 offen offset:2048  \n"
            "buffer_load_dwordx4  acc[124:127], %[v_os_b7], s[20:23], 0 offen offset:3072  \n"
            "s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
            "s_cselect_b32 s86, %[s_tile_os_b], 0      ; move b with cond \n"
            "s_add_u32     s20, s86, s20               ; move b with cond \n"
            "s_addc_u32    s21, 0, s21                 ; move b with cond \n"
            "s_waitcnt     vmcnt(40)                        \n"
            "s_barrier                                      \n"
            "ds_read_b128  v[64:67], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_0]\n"    // 1024: N stride, 64 K stride
            "ds_read_b128  v[68:71], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_1]\n"
            "ds_read_b128  v[72:75], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_2]\n"
            "ds_read_b128  v[76:79], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_3]\n"
            "ds_read_b128  v[80:83], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_4]\n"
            "ds_read_b128  v[84:87], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_5]\n"
            "ds_read_b128  v[88:91], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_6]\n"
            "ds_read_b128  v[92:95], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_7]\n"
            "L_start%=:                                                         \n"
            "  s_waitcnt     vmcnt(24) & lgkmcnt(0)                             \n"
            "  s_barrier                                                        \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[0:1], v[64:65], %[v_acc_0] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[2:3], v[66:67], %[v_acc_0] \n"
            "  buffer_load_dwordx4  acc[128:131], %[v_os_b0], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[4:5], v[68:69], %[v_acc_0] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[6:7], v[70:71], %[v_acc_0] \n"
            "  buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[8:9], v[72:73], %[v_acc_0] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[10:11], v[74:75], %[v_acc_0] \n"
            "  buffer_load_dwordx4  acc[132:135], %[v_os_b0], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[12:13], v[76:77], %[v_acc_0] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[14:15], v[78:79], %[v_acc_0] \n"
            "  buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[0:1], v[80:81], %[v_acc_1] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[2:3], v[82:83], %[v_acc_1] \n"
            "  buffer_load_dwordx4  acc[136:139], %[v_os_b0], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[4:5], v[84:85], %[v_acc_1] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[6:7], v[86:87], %[v_acc_1] \n"
            "  buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[8:9], v[88:89], %[v_acc_1] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[10:11], v[90:91], %[v_acc_1] \n"
            "  buffer_load_dwordx4  acc[140:143], %[v_os_b0], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[12:13], v[92:93], %[v_acc_1] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[14:15], v[94:95], %[v_acc_1] \n"
            "  buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[16:17], v[64:65], %[v_acc_2] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[18:19], v[66:67], %[v_acc_2] \n"
            "  buffer_load_dwordx4  acc[144:147], %[v_os_b1], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[20:21], v[68:69], %[v_acc_2] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[22:23], v[70:71], %[v_acc_2] \n"
            "  buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[24:25], v[72:73], %[v_acc_2] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[26:27], v[74:75], %[v_acc_2] \n"
            "  buffer_load_dwordx4  acc[148:151], %[v_os_b1], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[28:29], v[76:77], %[v_acc_2] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[30:31], v[78:79], %[v_acc_2] \n"
            "  buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[16:17], v[80:81], %[v_acc_3] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[18:19], v[82:83], %[v_acc_3] \n"
            "  buffer_load_dwordx4  acc[152:155], %[v_os_b1], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[20:21], v[84:85], %[v_acc_3] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[22:23], v[86:87], %[v_acc_3] \n"
            "  buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[24:25], v[88:89], %[v_acc_3] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[26:27], v[90:91], %[v_acc_3] \n"
            "  buffer_load_dwordx4  acc[156:159], %[v_os_b1], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[28:29], v[92:93], %[v_acc_3] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[30:31], v[94:95], %[v_acc_3] \n"
            "  buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[smem_sz], %[s_m0_init]                  \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[32:33], v[64:65], %[v_acc_4] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[34:35], v[66:67], %[v_acc_4] \n"
            "  buffer_load_dwordx4  acc[160:163], %[v_os_b2], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[36:37], v[68:69], %[v_acc_4] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[38:39], v[70:71], %[v_acc_4] \n"
            "  ds_read_b128  v[96:99], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_0]                \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[40:41], v[72:73], %[v_acc_4] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[42:43], v[74:75], %[v_acc_4] \n"
            "  buffer_load_dwordx4  acc[164:167], %[v_os_b2], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[44:45], v[76:77], %[v_acc_4] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[46:47], v[78:79], %[v_acc_4] \n"
            "  ds_read_b128  v[100:103], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_1]              \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[32:33], v[80:81], %[v_acc_5] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[34:35], v[82:83], %[v_acc_5] \n"
            "  buffer_load_dwordx4  acc[168:171], %[v_os_b2], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[36:37], v[84:85], %[v_acc_5] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[38:39], v[86:87], %[v_acc_5] \n"
            "  ds_read_b128  v[104:107], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_2]              \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[40:41], v[88:89], %[v_acc_5] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[42:43], v[90:91], %[v_acc_5] \n"
            "  buffer_load_dwordx4  acc[172:175], %[v_os_b2], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[44:45], v[92:93], %[v_acc_5] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[46:47], v[94:95], %[v_acc_5] \n"
            "  ds_read_b128  v[108:111], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_3]              \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[48:49], v[64:65], %[v_acc_6] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[50:51], v[66:67], %[v_acc_6] \n"
            "  buffer_load_dwordx4  acc[176:179], %[v_os_b3], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[52:53], v[68:69], %[v_acc_6] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[54:55], v[70:71], %[v_acc_6] \n"
            "  ds_read_b128  v[112:115], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_4]              \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[56:57], v[72:73], %[v_acc_6] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[58:59], v[74:75], %[v_acc_6] \n"
            "  buffer_load_dwordx4  acc[180:183], %[v_os_b3], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[60:61], v[76:77], %[v_acc_6] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[62:63], v[78:79], %[v_acc_6] \n"
            "  ds_read_b128  v[116:119], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_5]              \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[48:49], v[80:81], %[v_acc_7] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[50:51], v[82:83], %[v_acc_7] \n"
            "  buffer_load_dwordx4  acc[184:187], %[v_os_b3], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[52:53], v[84:85], %[v_acc_7] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[54:55], v[86:87], %[v_acc_7] \n"
            "  ds_read_b128  v[120:123], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_6]              \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[56:57], v[88:89], %[v_acc_7] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[58:59], v[90:91], %[v_acc_7] \n"
            "  buffer_load_dwordx4  acc[188:191], %[v_os_b3], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[60:61], v[92:93], %[v_acc_7] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[62:63], v[94:95], %[v_acc_7] \n"
            "  ds_read_b128  v[124:127], %[v_os_slda], offset:1*%[smem_sz] + %[sld_os_7]              \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[64:65], v[64:65], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[66:67], v[66:67], %[v_acc_8] \n"
            "  buffer_load_dwordx4  acc[192:195], %[v_os_b4], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[68:69], v[68:69], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[70:71], v[70:71], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[72:73], v[72:73], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[74:75], v[74:75], %[v_acc_8] \n"
            "  buffer_load_dwordx4  acc[196:199], %[v_os_b4], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[76:77], v[76:77], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[78:79], v[78:79], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[64:65], v[80:81], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[66:67], v[82:83], %[v_acc_9] \n"
            "  buffer_load_dwordx4  acc[200:203], %[v_os_b4], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[68:69], v[84:85], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[70:71], v[86:87], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[72:73], v[88:89], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[74:75], v[90:91], %[v_acc_9] \n"
            "  buffer_load_dwordx4  acc[204:207], %[v_os_b4], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[76:77], v[92:93], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[78:79], v[94:95], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[80:81], v[64:65], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[82:83], v[66:67], %[v_acc_10] \n"
            "  buffer_load_dwordx4  acc[208:211], %[v_os_b5], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[84:85], v[68:69], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[86:87], v[70:71], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[88:89], v[72:73], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[90:91], v[74:75], %[v_acc_10] \n"
            "  buffer_load_dwordx4  acc[212:215], %[v_os_b5], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[92:93], v[76:77], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[94:95], v[78:79], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[80:81], v[80:81], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[82:83], v[82:83], %[v_acc_11] \n"
            "  buffer_load_dwordx4  acc[216:219], %[v_os_b5], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[84:85], v[84:85], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[86:87], v[86:87], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[88:89], v[88:89], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[90:91], v[90:91], %[v_acc_11] \n"
            "  buffer_load_dwordx4  acc[220:223], %[v_os_b5], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[92:93], v[92:93], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[94:95], v[94:95], %[v_acc_11] \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[96:97], v[64:65], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[98:99], v[66:67], %[v_acc_12] \n"
            "  buffer_load_dwordx4  acc[224:227], %[v_os_b6], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[100:101], v[68:69], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[102:103], v[70:71], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[104:105], v[72:73], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[106:107], v[74:75], %[v_acc_12] \n"
            "  buffer_load_dwordx4  acc[228:231], %[v_os_b6], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[108:109], v[76:77], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[110:111], v[78:79], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[96:97], v[80:81], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[98:99], v[82:83], %[v_acc_13] \n"
            "  buffer_load_dwordx4  acc[232:235], %[v_os_b6], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[100:101], v[84:85], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[102:103], v[86:87], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[104:105], v[88:89], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[106:107], v[90:91], %[v_acc_13] \n"
            "  buffer_load_dwordx4  acc[236:239], %[v_os_b6], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[108:109], v[92:93], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[110:111], v[94:95], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[112:113], v[64:65], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[114:115], v[66:67], %[v_acc_14] \n"
            "  buffer_load_dwordx4  acc[240:243], %[v_os_b7], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[116:117], v[68:69], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[118:119], v[70:71], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[120:121], v[72:73], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[122:123], v[74:75], %[v_acc_14] \n"
            "  buffer_load_dwordx4  acc[244:247], %[v_os_b7], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[124:125], v[76:77], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[126:127], v[78:79], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[112:113], v[80:81], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[114:115], v[82:83], %[v_acc_15] \n"
            "  buffer_load_dwordx4  acc[248:251], %[v_os_b7], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[116:117], v[84:85], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[118:119], v[86:87], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[120:121], v[88:89], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[122:123], v[90:91], %[v_acc_15] \n"
            "  buffer_load_dwordx4  acc[252:255], %[v_os_b7], s[20:23], 0 offen offset:3072\n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[124:125], v[92:93], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[126:127], v[94:95], %[v_acc_15] \n"
            "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1                \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 0                                \n"
            "  s_cbranch_scc0 L_end%=                                       \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 2             ; move a with cond \n"
            "  s_cselect_b32 s86, %[s_tile_os_a], 0                          \n"
            "  s_add_u32     s16, s86, s16                                  \n"
            "  s_addc_u32    s17, 0, s17                                    \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
            "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
            "  s_add_u32     s20, s86, s20                                  \n"
            "  s_addc_u32    s21, 0, s21                                    \n"
            "  ;------------------------------------------                  \n"
            "  s_waitcnt     vmcnt(24) & lgkmcnt(0)                  \n"
            "  s_barrier                                             \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[128:129], v[96:97], %[v_acc_0] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[130:131], v[98:99], %[v_acc_0] \n"
            "  buffer_load_dwordx4  acc[0:3], %[v_os_b0], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[132:133], v[100:101], %[v_acc_0] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[134:135], v[102:103], %[v_acc_0] \n"
            "  buffer_load_dword   %[v_os_a0], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[136:137], v[104:105], %[v_acc_0] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[138:139], v[106:107], %[v_acc_0] \n"
            "  buffer_load_dwordx4  acc[4:7], %[v_os_b0], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[140:141], v[108:109], %[v_acc_0] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_0], acc[142:143], v[110:111], %[v_acc_0] \n"
            "  buffer_load_dword   %[v_os_a1], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[128:129], v[112:113], %[v_acc_1] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[130:131], v[114:115], %[v_acc_1] \n"
            "  buffer_load_dwordx4  acc[8:11], %[v_os_b0], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[132:133], v[116:117], %[v_acc_1] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[134:135], v[118:119], %[v_acc_1] \n"
            "  buffer_load_dword   %[v_os_a2], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[136:137], v[120:121], %[v_acc_1] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[138:139], v[122:123], %[v_acc_1] \n"
            "  buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[140:141], v[124:125], %[v_acc_1] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_1], acc[142:143], v[126:127], %[v_acc_1] \n"
            "  buffer_load_dword   %[v_os_a3], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[144:145], v[96:97], %[v_acc_2] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[146:147], v[98:99], %[v_acc_2] \n"
            "  buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[148:149], v[100:101], %[v_acc_2] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[150:151], v[102:103], %[v_acc_2] \n"
            "  buffer_load_dword   %[v_os_a4], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[152:153], v[104:105], %[v_acc_2] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[154:155], v[106:107], %[v_acc_2] \n"
            "  buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[156:157], v[108:109], %[v_acc_2] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_2], acc[158:159], v[110:111], %[v_acc_2] \n"
            "  buffer_load_dword   %[v_os_a5], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[144:145], v[112:113], %[v_acc_3] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[146:147], v[114:115], %[v_acc_3] \n"
            "  buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[148:149], v[116:117], %[v_acc_3] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[150:151], v[118:119], %[v_acc_3] \n"
            "  buffer_load_dword   %[v_os_a6], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, %[s_size_per_issue], m0                  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[152:153], v[120:121], %[v_acc_3] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[154:155], v[122:123], %[v_acc_3] \n"
            "  buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[156:157], v[124:125], %[v_acc_3] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_3], acc[158:159], v[126:127], %[v_acc_3] \n"
            "  buffer_load_dword   %[v_os_a7], s[16:19], 0 offen lds     \n"
            "  s_add_u32     m0, 0, %[s_m0_init]                  \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[160:161], v[96:97], %[v_acc_4] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[162:163], v[98:99], %[v_acc_4] \n"
            "  buffer_load_dwordx4  acc[32:35], %[v_os_b2], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[164:165], v[100:101], %[v_acc_4] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[166:167], v[102:103], %[v_acc_4] \n"
            "  ds_read_b128  v[64:67], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_0]  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[168:169], v[104:105], %[v_acc_4] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[170:171], v[106:107], %[v_acc_4] \n"
            "  buffer_load_dwordx4  acc[36:39], %[v_os_b2], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[172:173], v[108:109], %[v_acc_4] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_4], acc[174:175], v[110:111], %[v_acc_4] \n"
            "  ds_read_b128  v[68:71], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_1]  \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[160:161], v[112:113], %[v_acc_5] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[162:163], v[114:115], %[v_acc_5] \n"
            "  buffer_load_dwordx4  acc[40:43], %[v_os_b2], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[164:165], v[116:117], %[v_acc_5] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[166:167], v[118:119], %[v_acc_5] \n"
            "  ds_read_b128  v[72:75], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_2]                 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[168:169], v[120:121], %[v_acc_5] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[170:171], v[122:123], %[v_acc_5] \n"
            "  buffer_load_dwordx4  acc[44:47], %[v_os_b2], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[172:173], v[124:125], %[v_acc_5] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_5], acc[174:175], v[126:127], %[v_acc_5] \n"
            "  ds_read_b128  v[76:79], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_3]                \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[176:177], v[96:97], %[v_acc_6] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[178:179], v[98:99], %[v_acc_6] \n"
            "  buffer_load_dwordx4  acc[48:51], %[v_os_b3], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[180:181], v[100:101], %[v_acc_6] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[182:183], v[102:103], %[v_acc_6] \n"
            "  ds_read_b128  v[80:83], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_4]               \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[184:185], v[104:105], %[v_acc_6] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[186:187], v[106:107], %[v_acc_6] \n"
            "  buffer_load_dwordx4  acc[52:55], %[v_os_b3], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[188:189], v[108:109], %[v_acc_6] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_6], acc[190:191], v[110:111], %[v_acc_6] \n"
            "  ds_read_b128  v[84:87], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_5]            \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[176:177], v[112:113], %[v_acc_7] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[178:179], v[114:115], %[v_acc_7] \n"
            "  buffer_load_dwordx4  acc[56:59], %[v_os_b3], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[180:181], v[116:117], %[v_acc_7] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[182:183], v[118:119], %[v_acc_7] \n"
            "  ds_read_b128  v[88:91], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_6]                \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[184:185], v[120:121], %[v_acc_7] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[186:187], v[122:123], %[v_acc_7] \n"
            "  buffer_load_dwordx4  acc[60:63], %[v_os_b3], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[188:189], v[124:125], %[v_acc_7] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_7], acc[190:191], v[126:127], %[v_acc_7] \n"
            "  ds_read_b128  v[92:95], %[v_os_slda] offset:0*%[smem_sz] + %[sld_os_7]           \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[192:193], v[96:97], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[194:195], v[98:99], %[v_acc_8] \n"
            "  buffer_load_dwordx4  acc[64:67], %[v_os_b4], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[196:197], v[100:101], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[198:199], v[102:103], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[200:201], v[104:105], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[202:203], v[106:107], %[v_acc_8] \n"
            "  buffer_load_dwordx4  acc[68:71], %[v_os_b4], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[204:205], v[108:109], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_8], acc[206:207], v[110:111], %[v_acc_8] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[192:193], v[112:113], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[194:195], v[114:115], %[v_acc_9] \n"
            "  buffer_load_dwordx4  acc[72:75], %[v_os_b4], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[196:197], v[116:117], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[198:199], v[118:119], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[200:201], v[120:121], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[202:203], v[122:123], %[v_acc_9] \n"
            "  buffer_load_dwordx4  acc[76:79], %[v_os_b4], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[204:205], v[124:125], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_9], acc[206:207], v[126:127], %[v_acc_9] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[208:209], v[96:97], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[210:211], v[98:99], %[v_acc_10] \n"
            "  buffer_load_dwordx4  acc[80:83], %[v_os_b5], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[212:213], v[100:101], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[214:215], v[102:103], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[216:217], v[104:105], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[218:219], v[106:107], %[v_acc_10] \n"
            "  buffer_load_dwordx4  acc[84:87], %[v_os_b5], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[220:221], v[108:109], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_10], acc[222:223], v[110:111], %[v_acc_10] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[208:209], v[112:113], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[210:211], v[114:115], %[v_acc_11] \n"
            "  buffer_load_dwordx4  acc[88:91], %[v_os_b5], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[212:213], v[116:117], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[214:215], v[118:119], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[216:217], v[120:121], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[218:219], v[122:123], %[v_acc_11] \n"
            "  buffer_load_dwordx4  acc[92:95], %[v_os_b5], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[220:221], v[124:125], %[v_acc_11] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_11], acc[222:223], v[126:127], %[v_acc_11] \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[224:225], v[96:97], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[226:227], v[98:99], %[v_acc_12] \n"
            "  buffer_load_dwordx4  acc[96:99], %[v_os_b6], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[228:229], v[100:101], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[230:231], v[102:103], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[232:233], v[104:105], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[234:235], v[106:107], %[v_acc_12] \n"
            "  buffer_load_dwordx4  acc[100:103], %[v_os_b6], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[236:237], v[108:109], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_12], acc[238:239], v[110:111], %[v_acc_12] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[224:225], v[112:113], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[226:227], v[114:115], %[v_acc_13] \n"
            "  buffer_load_dwordx4  acc[104:107], %[v_os_b6], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[228:229], v[116:117], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[230:231], v[118:119], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[232:233], v[120:121], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[234:235], v[122:123], %[v_acc_13] \n"
            "  buffer_load_dwordx4  acc[108:111], %[v_os_b6], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[236:237], v[124:125], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_13], acc[238:239], v[126:127], %[v_acc_13] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[240:241], v[96:97], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[242:243], v[98:99], %[v_acc_14] \n"
            "  buffer_load_dwordx4  acc[112:115], %[v_os_b7], s[20:23], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[244:245], v[100:101], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[246:247], v[102:103], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[248:249], v[104:105], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[250:251], v[106:107], %[v_acc_14] \n"
            "  buffer_load_dwordx4  acc[116:119], %[v_os_b7], s[20:23], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[252:253], v[108:109], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_14], acc[254:255], v[110:111], %[v_acc_14] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[240:241], v[112:113], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[242:243], v[114:115], %[v_acc_15] \n"
            "  buffer_load_dwordx4  acc[120:123], %[v_os_b7], s[20:23], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[244:245], v[116:117], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[246:247], v[118:119], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[248:249], v[120:121], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[250:251], v[122:123], %[v_acc_15] \n"
            "  buffer_load_dwordx4  acc[124:127], %[v_os_b7], s[20:23], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[252:253], v[124:125], %[v_acc_15] \n"
            "  v_mfma_f32_16x16x16_bf16  %[v_acc_15], acc[254:255], v[126:127], %[v_acc_15] \n"
            "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1                \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 0                                \n"
            "  s_cbranch_scc0 L_end%=                                       \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 2             ; move a with cond \n"
            "  s_cselect_b32 s86, %[s_tile_os_a], 0                          \n"
            "  s_add_u32     s16, s86, s16                                  \n"
            "  s_addc_u32    s17, 0, s17                                    \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
            "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
            "  s_add_u32     s20, s86, s20                                  \n"
            "  s_addc_u32    s21, 0, s21                                    \n"
            "  s_branch     L_start%=                                       \n"
            "L_end%=:                                                       \n"
            ""
            :   [s_loop_cnt]"+s"(loop_cnt),
                [v_acc_0]"+v"(v_acc[0]),
                [v_acc_1]"+v"(v_acc[1]),
                [v_acc_2]"+v"(v_acc[2]),
                [v_acc_3]"+v"(v_acc[3]),
                [v_acc_4]"+v"(v_acc[4]),
                [v_acc_5]"+v"(v_acc[5]),
                [v_acc_6]"+v"(v_acc[6]),
                [v_acc_7]"+v"(v_acc[7]),
                [v_acc_8]"+v"(v_acc[8]),
                [v_acc_9]"+v"(v_acc[9]),
                [v_acc_10]"+v"(v_acc[10]),
                [v_acc_11]"+v"(v_acc[11]),
                [v_acc_12]"+v"(v_acc[12]),
                [v_acc_13]"+v"(v_acc[13]),
                [v_acc_14]"+v"(v_acc[14]),
                [v_acc_15]"+v"(v_acc[15]),
                [s_mem_]"+r"(smem)
            : [s_res_a0]"s"(res_a[0]),
                [s_res_a1]"s"(res_a[1]),
                [s_res_a2]"s"(res_a[2]),
                [s_res_a3]"s"(res_a[3]),
                [s_res_b0]"s"(res_b[0]),
                [s_res_b1]"s"(res_b[1]),
                [s_res_b2]"s"(res_b[2]),
                [s_res_b3]"s"(res_b[3]),
                [v_os_a0]"v"(static_cast<index_t>(cached_coords_a[number<0>{}] * sizeof(bf16_t))),
                [v_os_a1]"v"(static_cast<index_t>(cached_coords_a[number<1>{}] * sizeof(bf16_t))),
                [v_os_a2]"v"(static_cast<index_t>(cached_coords_a[number<2>{}] * sizeof(bf16_t))),
                [v_os_a3]"v"(static_cast<index_t>(cached_coords_a[number<3>{}] * sizeof(bf16_t))),
                [v_os_a4]"v"(static_cast<index_t>(cached_coords_a[number<4>{}] * sizeof(bf16_t))),
                [v_os_a5]"v"(static_cast<index_t>(cached_coords_a[number<5>{}] * sizeof(bf16_t))),
                [v_os_a6]"v"(static_cast<index_t>(cached_coords_a[number<6>{}] * sizeof(bf16_t))),
                [v_os_a7]"v"(static_cast<index_t>(cached_coords_a[number<7>{}] * sizeof(bf16_t))),

                [v_os_b0]"v"(static_cast<index_t>(cached_coords_b[number<0>{}] * sizeof(bf16_t))),
                [v_os_b1]"v"(static_cast<index_t>(cached_coords_b[number<1>{}] * sizeof(bf16_t))),
                [v_os_b2]"v"(static_cast<index_t>(cached_coords_b[number<2>{}] * sizeof(bf16_t))),
                [v_os_b3]"v"(static_cast<index_t>(cached_coords_b[number<3>{}] * sizeof(bf16_t))),
                [v_os_b4]"v"(static_cast<index_t>(cached_coords_b[number<4>{}] * sizeof(bf16_t))),
                [v_os_b5]"v"(static_cast<index_t>(cached_coords_b[number<5>{}] * sizeof(bf16_t))),
                [v_os_b6]"v"(static_cast<index_t>(cached_coords_b[number<6>{}] * sizeof(bf16_t))),
                [v_os_b7]"v"(static_cast<index_t>(cached_coords_b[number<7>{}] * sizeof(bf16_t))),

                [v_os_slda]"v"(static_cast<index_t>(a_sld.cached_coords_[number<0>{}].get_offset() * sizeof(bf16_t))),
                [s_m0_init]"s"(m0_init_value),
                [s_size_per_issue]"s"(size_per_issue),
                [smem_sz]"n"(smem_buf_size),  //(smem_buf_size),
                [sld_os_0]"n"(sld_os[number<0>{}].value),
                [sld_os_1]"n"(sld_os[number<1>{}].value),
                [sld_os_2]"n"(sld_os[number<2>{}].value),
                [sld_os_3]"n"(sld_os[number<3>{}].value),
                [sld_os_4]"n"(sld_os[number<4>{}].value),
                [sld_os_5]"n"(sld_os[number<5>{}].value),
                [sld_os_6]"n"(sld_os[number<6>{}].value),
                [sld_os_7]"n"(sld_os[number<7>{}].value),
                [s_tile_os_a]"s"(tile_offset_a_bytes),
                [s_tile_os_b]"s"(tile_offset_b_bytes)
            : "memory", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
          "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19",
          "a20", "a21", "a22", "a23", "a24", "a25", "a26", "a27", "a28", "a29",
          "a30", "a31", "a32", "a33", "a34", "a35", "a36", "a37", "a38", "a39",
          "a40", "a41", "a42", "a43", "a44", "a45", "a46", "a47", "a48", "a49",
          "a50", "a51", "a52", "a53", "a54", "a55", "a56", "a57", "a58", "a59",
          "a60", "a61", "a62", "a63", "a64", "a65", "a66", "a67", "a68", "a69",
          "a70", "a71", "a72", "a73", "a74", "a75", "a76", "a77", "a78", "a79",
          "a80", "a81", "a82", "a83", "a84", "a85", "a86", "a87", "a88", "a89",
          "a90", "a91", "a92", "a93", "a94", "a95", "a96", "a97", "a98", "a99",
          "a100", "a101", "a102", "a103", "a104", "a105", "a106", "a107",
          "a108", "a109", "a110", "a111", "a112", "a113", "a114", "a115",
          "a116", "a117", "a118", "a119", "a120", "a121", "a122", "a123",
          "a124", "a125", "a126", "a127", "a128", "a129", "a130", "a131",
          "a132", "a133", "a134", "a135", "a136", "a137", "a138", "a139",
          "a140", "a141", "a142", "a143", "a144", "a145", "a146", "a147",
          "a148", "a149", "a150", "a151", "a152", "a153", "a154", "a155",
          "a156", "a157", "a158", "a159", "a160", "a161", "a162", "a163",
          "a164", "a165", "a166", "a167", "a168", "a169", "a170", "a171",
          "a172", "a173", "a174", "a175", "a176", "a177", "a178", "a179",
          "a180", "a181", "a182", "a183", "a184", "a185", "a186", "a187",
          "a188", "a189", "a190", "a191", "a192", "a193", "a194", "a195",
          "a196", "a197", "a198", "a199", "a200", "a201", "a202", "a203",
          "a204", "a205", "a206", "a207", "a208", "a209", "a210", "a211",
          "a212", "a213", "a214", "a215", "a216", "a217", "a218", "a219",
          "a220", "a221", "a222", "a223", "a224", "a225", "a226", "a227",
          "a228", "a229", "a230", "a231", "a232", "a233", "a234", "a235",
          "a236", "a237", "a238", "a239", "a240", "a241", "a242", "a243",
          "a244", "a245", "a246", "a247", "a248", "a249", "a250", "a251",
          "a252", "a253", "a254", "a255", 
          "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23",
          "s86",    // s86 as tmp
          "v64", "v65", "v66", "v67", "v68", "v69",
          "v70", "v71", "v72", "v73", "v74", "v75", "v76", "v77", "v78", "v79",
          "v80", "v81", "v82", "v83", "v84", "v85", "v86", "v87", "v88", "v89",
          "v90", "v91", "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99",
          "v100", "v101", "v102", "v103", "v104", "v105", "v106", "v107",
          "v108", "v109", "v110", "v111", "v112", "v113", "v114", "v115",
          "v116", "v117", "v118", "v119", "v120", "v121", "v122", "v123",
          "v124", "v125", "v126", "v127"
        );
        _Pragma("clang diagnostic pop");
        // clang-format on
        (void)smem_buf_size;
        (void)sld_os;
        // return local scratch
        auto c = MakeCBlockTile();
        for(auto i = 0; i < 16; i++)
        {
            c.get_thread_buffer()[4 * i + 0] = v_acc[i].x;
            c.get_thread_buffer()[4 * i + 1] = v_acc[i].y;
            c.get_thread_buffer()[4 * i + 2] = v_acc[i].z;
            c.get_thread_buffer()[4 * i + 3] = v_acc[i].w;
        }
        return c;
    }
};

} // namespace ck_tile
