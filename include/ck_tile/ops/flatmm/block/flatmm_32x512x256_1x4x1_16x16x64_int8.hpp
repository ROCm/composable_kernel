// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/flatmm/block/flatmm_uk_config.hpp"

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
struct Flatmm_32x512x256_1x4x1_16x16x64_Base // for int8/fp8
{
    static constexpr index_t Block_M = 32;
    static constexpr index_t Block_N = 512;
    static constexpr index_t Block_K = 256;

    static constexpr index_t WarpPerBlock_M = 1;
    static constexpr index_t WarpPerBlock_N = 4;
    static constexpr index_t WarpPerBlock_K = 1;

    static constexpr index_t NumWarps = 4;

    static constexpr index_t Warp_M = 16;
    static constexpr index_t Warp_N = 16;
    static constexpr index_t Warp_K = 64; // 32 * SubKPacks

    static constexpr index_t BlockSize = 256;

    static constexpr index_t SubKPacks = 4; // this is used to gurantee every threads can do dwordx4

    // TODO: note Nr/Kr/W need consider SubKPacks
    static constexpr index_t Block_W  = Warp_N * Warp_K;  // 1024 element
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

        using WG = WarpGemmMfma_i32_16x16x64_int8_int8_CTransposed;

        constexpr auto c_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            c_block_outer_dstr_encoding, typename WG::CWarpDstrEncoding{});
        constexpr auto c_block_dstr = make_static_tile_distribution(c_block_dstr_encode);
        return c_block_dstr;
    }

    static CK_TILE_DEVICE constexpr auto MakeCBlockTile()
    {
        using CDataType             = float;
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

        constexpr index_t KPack_  = 16;      // GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t KVector = 4;      // GetAlignment_A<Problem>(); // async copy 1 dword
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
        constexpr index_t KPack_ = 16;      // GetSmemKPack_A<Problem>(); // LDS
        constexpr index_t KPad   = KPack_; // pad between warps

        constexpr index_t kAMLane     = 16;
        constexpr index_t kABKLane    = 4;
        constexpr index_t kABKPerLane = 8;
        constexpr index_t kKIter      = 2;
        static_assert(KPack_ == (kABKPerLane * kKIter));

        constexpr auto lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(number<Repeat_M>{}, // m0 y
                                                    number<kAMLane>{},  // m1 p
                                                    number<Repeat_K>{}, // k0 y
                                                    number<kABKLane>{}, // k1 p
                                                    number<KPack_>{}),  // k2 y-vector
                                         make_tuple(number<kAMLane*(Block_K + KPad)>{}, // m0
                                                    number<Block_K + KPad>{},           // m1
                                                    number<kABKLane * KPack_>{},        // k0
                                                    number<KPack_>{},                   // k1
                                                    number<1>{}),                       // k2
                                         number<KPack_>{}, // lds load vector
                                         number<1>{});

        constexpr auto lds_desc_m_k = transform_tensor_descriptor(
            lds_block_desc_0,
            make_tuple(make_merge_transform(make_tuple(number<Repeat_M>{}, number<kAMLane>{})),
                       make_merge_transform(
                           make_tuple(number<Repeat_K>{}, number<kABKLane>{}, number<KPack_>{}))),
            make_tuple(sequence<0, 1>{}, sequence<2, 3, 4>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return lds_desc_m_k;
    }

    static constexpr auto GetGemm_AWarpEnc()
    {
        constexpr index_t kAMLane     = 16;
        constexpr index_t kABKLane    = 4;
        constexpr index_t kABKPerLane = 8;
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
        return 32 * (256 + 16) * sizeof(int8_t);
    }
};

struct Flatmm_32x512x256_1x4x1_16x16x64_int8 : public Flatmm_32x512x256_1x4x1_16x16x64_Base
{
    using ADataType = int8_t;
    using BDataType = int8_t;

    // TODO: need paired with tile_window_linear!
    // TODO: need call init_raw() before call this function!
    template <typename Ascale, typename GQscale,  typename ARes, typename ACoords, typename BRes, typename BCoords>
    CK_TILE_DEVICE auto
    operator()( const Ascale& a_scale_,
                const GQscale& gq_scale_,
                const ARes& res_a,
               const ACoords& cached_coords_a,
               const BRes& res_b,
               const BCoords& cached_coords_b,
               CK_TILE_LDS_ADDR void* smem,
               index_t k,
               index_t tile_offset_a, // for each tile, the offset to move for each unroll
               index_t tile_offset_b) // for each tile, the offset to move for each unroll
    {
        static_assert(ACoords::size() == Block_M * Block_K / BlockSize / 4 /*2x per dword*/); // 8
        static_assert(BCoords::size() == Repeat_N);
        static_assert(Ascale::size() == Repeat_M);

        auto a_sst = make_tile_window(
            make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem), MakeLdsStoreDesc_A()),
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
                    reinterpret_cast<CK_TILE_LDS_ADDR ADataType*>(smem), MakeLdsLoadDesc_A()),
                MakeLdsLoadDesc_A().get_lengths(),
                {0, 0},
                make_static_tile_distribution(a_block_dstr_encode));
        }();

        const index_t tile_offset_a_bytes = tile_offset_a * sizeof(ADataType);
        const index_t tile_offset_b_bytes = tile_offset_b * sizeof(BDataType);

        const auto [m0_init_value, size_per_issue] = get_async_store_smem_info(a_sst);
        constexpr auto smem_buf_size =
            MakeLdsLoadDesc_A().get_element_space_size() * sizeof(ADataType);
        static_assert(a_sld.get_num_of_access() == 8);
        constexpr auto sld_os = generate_tuple(
            [&](auto i_access) {
                return number<a_sld.get_bottom_linear_offset(i_access) * sizeof(ADataType)>{};
            },
            number<a_sld.get_num_of_access()>{});

        //index_t loop_cnt = k / Block_K;
        index_t loop_cnt = k;

        // this is the acc thread buffer
	    register int v_z0 asm("v128") = 0;
        register int v_z1 asm("v129") = 0;
        register int v_z2 asm("v130") = 0;
        register int v_z3 asm("v131") = 0;
        register int v_z4 asm("v132") = 0;
        register int v_z5 asm("v133") = 0;
        register int v_z6 asm("v134") = 0;
        register int v_z7 asm("v135") = 0;
        register int v_z8 asm("v136") = 0;
        register int v_z9 asm("v137") = 0;
        register int v_z10 asm("v138") = 0;
        register int v_z11 asm("v139") = 0;
        register int v_z12 asm("v140") = 0;
        register int v_z13 asm("v141") = 0;
        register int v_z14 asm("v142") = 0;
        register int v_z15 asm("v143") = 0;
        register int v_z16 asm("v144") = 0;
        register int v_z17 asm("v145") = 0;
        register int v_z18 asm("v146") = 0;
        register int v_z19 asm("v147") = 0;
        register int v_z20 asm("v148") = 0;
        register int v_z21 asm("v149") = 0;
        register int v_z22 asm("v150") = 0;
        register int v_z23 asm("v151") = 0;
        register int v_z24 asm("v152") = 0;
        register int v_z25 asm("v153") = 0;
        register int v_z26 asm("v154") = 0;
        register int v_z27 asm("v155") = 0;
        register int v_z28 asm("v156") = 0;
        register int v_z29 asm("v157") = 0;
        register int v_z30 asm("v158") = 0;
        register int v_z31 asm("v159") = 0;
        register int v_z32 asm("v160") = 0;
        register int v_z33 asm("v161") = 0;
        register int v_z34 asm("v162") = 0;
        register int v_z35 asm("v163") = 0;
        register int v_z36 asm("v164") = 0;
        register int v_z37 asm("v165") = 0;
        register int v_z38 asm("v166") = 0;
        register int v_z39 asm("v167") = 0;
        register int v_z40 asm("v168") = 0;
        register int v_z41 asm("v169") = 0;
        register int v_z42 asm("v170") = 0;
        register int v_z43 asm("v171") = 0;
        register int v_z44 asm("v172") = 0;
        register int v_z45 asm("v173") = 0;
        register int v_z46 asm("v174") = 0;
        register int v_z47 asm("v175") = 0;
        register int v_z48 asm("v176") = 0;
        register int v_z49 asm("v177") = 0;
        register int v_z50 asm("v178") = 0;
        register int v_z51 asm("v179") = 0;
        register int v_z52 asm("v180") = 0;
        register int v_z53 asm("v181") = 0;
        register int v_z54 asm("v182") = 0;
        register int v_z55 asm("v183") = 0;
        register int v_z56 asm("v184") = 0;
        register int v_z57 asm("v185") = 0;
        register int v_z58 asm("v186") = 0;
        register int v_z59 asm("v187") = 0;
        register int v_z60 asm("v188") = 0;
        register int v_z61 asm("v189") = 0;
        register int v_z62 asm("v190") = 0;
        register int v_z63 asm("v191") = 0;	
        // B nr->kr
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Winline-asm"
        // clang-format off
        asm volatile(
#define CK_TILE_FLATMM_UK_MFMA CK_TILE_FLATMM_UK_MFMA_INT8
#include "uk/flatmm_uk_gfx9_32x512x256_1x1x1_16x16x32_int8.inc"
#undef CK_TILE_FLATMM_UK_MFMA
            :   [s_loop_cnt]"+s"(loop_cnt),
                [c0]"+v"(v_z0),
                [c1]"+v"(v_z1),
                [c2]"+v"(v_z2),
                [c3]"+v"(v_z3),
                [c4]"+v"(v_z4),
                [c5]"+v"(v_z5),
                [c6]"+v"(v_z6),
                [c7]"+v"(v_z7),
                [c8]"+v"(v_z8),
                [c9]"+v"(v_z9),
                [c10]"+v"(v_z10),
                [c11]"+v"(v_z11),
                [c12]"+v"(v_z12),
                [c13]"+v"(v_z13),
                [c14]"+v"(v_z14),
                [c15]"+v"(v_z15),
                [c16]"+v"(v_z16),
                [c17]"+v"(v_z17),
                [c18]"+v"(v_z18),
                [c19]"+v"(v_z19),
                [c20]"+v"(v_z20),
                [c21]"+v"(v_z21),
                [c22]"+v"(v_z22),
                [c23]"+v"(v_z23),
                [c24]"+v"(v_z24),
                [c25]"+v"(v_z25),
                [c26]"+v"(v_z26),
                [c27]"+v"(v_z27),
                [c28]"+v"(v_z28),
                [c29]"+v"(v_z29),
                [c30]"+v"(v_z30),
                [c31]"+v"(v_z31),
                [c32]"+v"(v_z32),
                [c33]"+v"(v_z33),
                [c34]"+v"(v_z34),
                [c35]"+v"(v_z35),
                [c36]"+v"(v_z36),
                [c37]"+v"(v_z37),
                [c38]"+v"(v_z38),
                [c39]"+v"(v_z39),
                [c40]"+v"(v_z40),
                [c41]"+v"(v_z41),
                [c42]"+v"(v_z42),
                [c43]"+v"(v_z43),
                [c44]"+v"(v_z44),
                [c45]"+v"(v_z45),
                [c46]"+v"(v_z46),
                [c47]"+v"(v_z47),
                [c48]"+v"(v_z48),
                [c49]"+v"(v_z49),
                [c50]"+v"(v_z50),
                [c51]"+v"(v_z51),
                [c52]"+v"(v_z52),
                [c53]"+v"(v_z53),
                [c54]"+v"(v_z54),
                [c55]"+v"(v_z55),
                [c56]"+v"(v_z56),
                [c57]"+v"(v_z57),
                [c58]"+v"(v_z58),
                [c59]"+v"(v_z59),
                [c60]"+v"(v_z60),
                [c61]"+v"(v_z61),
                [c62]"+v"(v_z62),
                [c63]"+v"(v_z63),
                [s_mem_]"+r"(smem)
            :   [a_scale0]"v"(a_scale_[0]),
                [a_scale1]"v"(a_scale_[1]),
                [gq_scale0]"v"(gq_scale_[0]),
                [gq_scale1]"v"(gq_scale_[1]),
                [s_res_a]"s"(res_a),
                // [s_res_a1]"s"(res_a[1]),
                // [s_res_a2]"s"(res_a[2]),
                // [s_res_a3]"s"(res_a[3]),
                [s_res_b]"s"(res_b),
                // [s_res_b1]"s"(res_b[1]),
                // [s_res_b2]"s"(res_b[2]),
                // [s_res_b3]"s"(res_b[3]),
                [v_os_a0]"v"(static_cast<index_t>(cached_coords_a[number<0>{}] * sizeof(ADataType))),
                [v_os_a1]"v"(static_cast<index_t>(cached_coords_a[number<1>{}] * sizeof(ADataType))),
                [v_os_a2]"v"(static_cast<index_t>(cached_coords_a[number<2>{}] * sizeof(ADataType))),
                [v_os_a3]"v"(static_cast<index_t>(cached_coords_a[number<3>{}] * sizeof(ADataType))),
                [v_os_a4]"v"(static_cast<index_t>(cached_coords_a[number<4>{}] * sizeof(ADataType))),
                [v_os_a5]"v"(static_cast<index_t>(cached_coords_a[number<5>{}] * sizeof(ADataType))),
                [v_os_a6]"v"(static_cast<index_t>(cached_coords_a[number<6>{}] * sizeof(ADataType))),
                [v_os_a7]"v"(static_cast<index_t>(cached_coords_a[number<7>{}] * sizeof(ADataType))),

                [v_os_b0]"v"(static_cast<index_t>(cached_coords_b[number<0>{}] * sizeof(BDataType))),
                [v_os_b1]"v"(static_cast<index_t>(cached_coords_b[number<1>{}] * sizeof(BDataType))),
                [v_os_b2]"v"(static_cast<index_t>(cached_coords_b[number<2>{}] * sizeof(BDataType))),
                [v_os_b3]"v"(static_cast<index_t>(cached_coords_b[number<3>{}] * sizeof(BDataType))),
                [v_os_b4]"v"(static_cast<index_t>(cached_coords_b[number<4>{}] * sizeof(BDataType))),
                [v_os_b5]"v"(static_cast<index_t>(cached_coords_b[number<5>{}] * sizeof(BDataType))),
                [v_os_b6]"v"(static_cast<index_t>(cached_coords_b[number<6>{}] * sizeof(BDataType))),
                [v_os_b7]"v"(static_cast<index_t>(cached_coords_b[number<7>{}] * sizeof(BDataType))),

                [v_os_sld]"v"(static_cast<index_t>(a_sld.cached_coords_[number<0>{}].get_offset() * sizeof(ADataType))),
                [s_m0_init]"s"(m0_init_value),
                [s_size_per_issue]"s"(size_per_issue),
                [smem_sz]"n"(smem_buf_size),  //(smem_buf_size),
                [s_wave_id]"s"(get_warp_id()),
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
            : 
         "memory", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
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
          "s6", "s7", "s40", "s41", "s42", "s43", "s44", "s45",
          "s46", "s47", "s48", "s49", "s50", "s51", "s52", "s53", "s54",
          "s55", "s56", "s57", "s58", "s59", "s60", "s61", "s62", "s63",
          "s64", "s65", "s66", "s67", "s68", "s69", "s70", "s71", "s72",
          "s73", "s74", "s75", "s76", "s77", "s78", "s79", "s80",    // s86 as tmp
          "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", 
          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
          "v20", "v21", "v22", "v23", "v24", "v25", "v50", "v51", "v52", "v53", "v54", "v55",
          "v56", "v57",  "v64",
          "v65", "v66", "v67", "v68", "v69", "v70", "v71", "v72", "v73",
          "v74", "v75", "v76", "v77", "v78", "v79", "v80", "v81", "v82",
          "v83", "v84", "v85", "v86", "v87", "v88", "v89", "v90", "v91",
          "v92", "v93", "v94", "v95", "v96", "v97", "v98", "v99", "v100",
          "v101", "v102", "v103", "v104", "v105", "v106", "v107", "v108",
          "v109", "v110", "v111", "v112", "v113", "v114", "v115", "v116",
          "v117", "v118", "v119", "v120", "v121", "v122", "v123", "v124",
          "v125", "v126", "v127", "v128", "v129", "v130", "v131", "v132",
          "v133", "v134", "v135", "v136", "v137", "v138", "v139", "v140",
          "v141", "v142", "v143", "v144", "v145", "v146", "v147", "v148",
          "v149", "v150", "v151", "v152", "v153", "v154", "v155", "v156",
          "v157", "v158", "v159", "v160", "v161", "v162", "v163", "v164",
          "v165", "v166", "v167", "v168", "v169", "v170", "v171", "v172",
          "v173", "v174", "v175", "v176", "v177", "v178", "v179", "v180",
          "v181", "v182", "v183", "v184", "v185", "v186", "v187", "v188",
          "v189", "v190", "v191", "v192", "v193", "v194", "v195", "v196",
          "v197", "v198", "v199", "v200", "v201", "v202", "v203", "v204",
          "v205", "v206", "v207", "v208", "v209", "v210", "v211", "v212",
          "v213", "v214", "v215", "v216", "v217", "v218", "v219", "v220",
          "v221", "v222", "v223", "v224", "v225", "v226", "v227", "v228",
          "v229", "v230", "v231", "v232", "v233", "v234", "v235", "v236",
          "v237", "v238", "v239", "v240", "v241", "v242", "v243", "v244",
          "v245", "v246", "v247", "v248", "v249", "v250", "v251", "v252",
          "v253", "v254", "v255"
        );
        // clang-format on
#pragma clang diagnostic pop
        
			
        // return local scratch
        auto c = MakeCBlockTile();
            c.get_thread_buffer()[0] = v_z0;
            c.get_thread_buffer()[1] = v_z1;
            c.get_thread_buffer()[2] = v_z2;
            c.get_thread_buffer()[3] = v_z3;
            c.get_thread_buffer()[4] = v_z4;
            c.get_thread_buffer()[5] = v_z5;
            c.get_thread_buffer()[6] = v_z6;
            c.get_thread_buffer()[7] = v_z7;
            c.get_thread_buffer()[8] = v_z8;
            c.get_thread_buffer()[9] = v_z9;
            c.get_thread_buffer()[10] = v_z10;
            c.get_thread_buffer()[11] = v_z11;
            c.get_thread_buffer()[12] = v_z12;
            c.get_thread_buffer()[13] = v_z13;
            c.get_thread_buffer()[14] = v_z14;
            c.get_thread_buffer()[15] = v_z15;
            c.get_thread_buffer()[16] = v_z16;
            c.get_thread_buffer()[17] = v_z17;
            c.get_thread_buffer()[18] = v_z18;
            c.get_thread_buffer()[19] = v_z19;
            c.get_thread_buffer()[20] = v_z20;
            c.get_thread_buffer()[21] = v_z21;
            c.get_thread_buffer()[22] = v_z22;
            c.get_thread_buffer()[23] = v_z23;
            c.get_thread_buffer()[24] = v_z24;
            c.get_thread_buffer()[25] = v_z25;
            c.get_thread_buffer()[26] = v_z26;
            c.get_thread_buffer()[27] = v_z27;
            c.get_thread_buffer()[28] = v_z28;
            c.get_thread_buffer()[29] = v_z29;
            c.get_thread_buffer()[30] = v_z30;
            c.get_thread_buffer()[31] = v_z31;
            c.get_thread_buffer()[32] = v_z32;
            c.get_thread_buffer()[33] = v_z33;
            c.get_thread_buffer()[34] = v_z34;
            c.get_thread_buffer()[35] = v_z35;
            c.get_thread_buffer()[36] = v_z36;
            c.get_thread_buffer()[37] = v_z37;
            c.get_thread_buffer()[38] = v_z38;
            c.get_thread_buffer()[39] = v_z39;
            c.get_thread_buffer()[40] = v_z40;
            c.get_thread_buffer()[41] = v_z41;
            c.get_thread_buffer()[42] = v_z42;
            c.get_thread_buffer()[43] = v_z43;
            c.get_thread_buffer()[44] = v_z44;
            c.get_thread_buffer()[45] = v_z45;
            c.get_thread_buffer()[46] = v_z46;
            c.get_thread_buffer()[47] = v_z47;
            c.get_thread_buffer()[48] = v_z48;
            c.get_thread_buffer()[49] = v_z49;
            c.get_thread_buffer()[50] = v_z50;
            c.get_thread_buffer()[51] = v_z51;
            c.get_thread_buffer()[52] = v_z52;
            c.get_thread_buffer()[53] = v_z53;
            c.get_thread_buffer()[54] = v_z54;
            c.get_thread_buffer()[55] = v_z55;
            c.get_thread_buffer()[56] = v_z56;
            c.get_thread_buffer()[57] = v_z57;
            c.get_thread_buffer()[58] = v_z58;
            c.get_thread_buffer()[59] = v_z59;
            c.get_thread_buffer()[60] = v_z60;
            c.get_thread_buffer()[61] = v_z61;
            c.get_thread_buffer()[62] = v_z62;
            c.get_thread_buffer()[63] = v_z63;
        return c;
    }
};

} // namespace ck_tile
