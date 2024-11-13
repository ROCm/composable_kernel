// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"

namespace ck_tile {

// "S"tream update output along "N"
// A in smem, B load from global
// require 4 wave, occupancy=1c
struct FlatmmSnUK_GFX9_32x128x512_1x4x1_16x16x16_BF16
{
    static constexpr index_t Block_M = 32;
    static constexpr index_t Block_N = 128;
    static constexpr index_t Block_K = 512;

    static constexpr index_t WarpPerBlock_M = 1;
    static constexpr index_t WarpPerBlock_N = 4;
    static constexpr index_t WarpPerBlock_K = 1;

    static constexpr index_t Warp_M = 16;
    static constexpr index_t Warp_N = 16;
    static constexpr index_t Warp_K = 16;

    static constexpr index_t BlockSize = 256;

    static constexpr index_t KPack = 2; // this is used to gurantee every threads can do dwordx4

    // TODO: note Nr/Kr/W need consider KPack
    static constexpr index_t Block_W  = Warp_N * Warp_K * KPack;    // 512 element
    static constexpr index_t Block_Nr = Block_N / Warp_N;           // 32 element, 4 per wave
    static constexpr index_t Block_Kr = Block_K / (Warp_K * KPack); // 4

    static constexpr index_t Repeat_M = Block_M / (Warp_M * WarpPerBlock_M); // 2
    static constexpr index_t Repeat_N = Block_N / (Warp_N * WarpPerBlock_N); // 8
    static constexpr index_t Repeat_K = Block_K / (Warp_K * WarpPerBlock_K); // 8

    using BDataType = bf16_t;
    using ODataType = bf16_t;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        //                    y     y     p     p      p      y
        // reg before shfl  M0(2)*N0(2)*Nl(4)*Nw(4)*Mw(16)*Nv(4)
        // but order is N0*M0*Nv
        // in LDS we need store as
        //          M0(2)* N0(2) *  Nl(4) * Nw(4) * (Mw(16)*Nv(4) + 4)
        //             y    y       wave-id  lid/16  lid%16   v
        return 2 * 2 * 4 * 4 * (16 * 4 + 4) * sizeof(bf16_t);
    }

    // TODO: need paired with tile_window_linear!
    // TODO: need call init_raw() before call this function!
    // template <typename AWindow, typename BWindow, typename OWindow, typename ScaleTensor>
    template <typename BRes,
              typename BCoords,
              typename ORes,
              typename OCoords,
              typename OFlags,
              typename ScaleTensor>
    CK_TILE_DEVICE auto
    operator()(const BRes& res_b,
               const BCoords& cached_coords_b,
               const ORes& res_o,
               const OCoords& cached_coords_o,
               const OFlags& o_flags, // this should be in sgpr
               CK_TILE_LDS_ADDR void* smem,
               // OWindow& o_window_,
               index_t n, // loop along n dim
               const ScaleTensor& scale_,
               index_t tile_offset_b, // stride b is fixed to blockKr * blockW, but still can adjust
               index_t tile_offset_o)
    {
        static_assert(BCoords::size() == 8); // 8
        static_assert(OCoords::size() == 8);

        const index_t tile_stride_b_bytes = tile_offset_b * sizeof(BDataType);
        const index_t tile_stride_o_bytes = tile_offset_o * sizeof(ODataType);

        static_assert(ScaleTensor::size() == 2);
        float s0 = scale_[number<0>{}];
        float s1 = scale_[number<1>{}];

        index_t loop_cnt = n / Block_N;

        register float v_c0 asm("v64");
        register float v_c1 asm("v65");
        register float v_c2 asm("v66");
        register float v_c3 asm("v67");
        register float v_c4 asm("v68");
        register float v_c5 asm("v69");
        register float v_c6 asm("v70");
        register float v_c7 asm("v71");
        register float v_c8 asm("v72");
        register float v_c9 asm("v73");
        register float v_c10 asm("v74");
        register float v_c11 asm("v75");
        register float v_c12 asm("v76");
        register float v_c13 asm("v77");
        register float v_c14 asm("v78");
        register float v_c15 asm("v79");
        register float v_c16 asm("v80");
        register float v_c17 asm("v81");
        register float v_c18 asm("v82");
        register float v_c19 asm("v83");
        register float v_c20 asm("v84");
        register float v_c21 asm("v85");
        register float v_c22 asm("v86");
        register float v_c23 asm("v87");
        register float v_c24 asm("v88");
        register float v_c25 asm("v89");
        register float v_c26 asm("v90");
        register float v_c27 asm("v91");
        register float v_c28 asm("v92");
        register float v_c29 asm("v93");
        register float v_c30 asm("v94");
        register float v_c31 asm("v95");
        int32_t nan_hi = 0x7fff0000;
        int32_t nan_lo = 0x00007fff;

        // in smem, the layout is  M0(2)*K0(128)*M1(16)*K1(4)
        // every threads need 8xK in contiguous register
        // ... and every wave need the same data
        int lane_id  = threadIdx.x % 64;
        int sld_y_os = (lane_id % 16) * 4 + (lane_id / 16) * 128;
        sld_y_os *= 2;

        //                    y     y     p     p      p      y
        // reg before shfl  M0(2)*N0(2)*Nl(4)*Nw(4)*Mw(16)*Nv(4)
        // but order is N0*M0*Nv
        // in LDS we need store as
        //          M0(2)* N0(2) *  Nl(4) * Nw(4) * (Mw(16)*Nv(4) + 4)
        //             y    y       wave-id  lid/16  lid%16   v
        int sfl_sst = (threadIdx.x % 16 * 4 + 4) * (threadIdx.x / 16);
        sfl_sst *= 2;

        // from LDS we need load as
        //          M0(2)*    N0(2) *  Nl(4) * Nw(4) * (Mw(16)         *  Nv(4) + 4)
        //        ( 2 issue)    (rem 32-lane)        (4 wave*4issue)   2lane*1ussue(pk2)
        int sfl_sld = (lane_id % 2) * 2 + (lane_id / 2) * (64 + 4) + (threadIdx.x / 64) * 4;
        sfl_sld *= 2;

        // B nr->kr
        // clang-format off
        _Pragma("clang diagnostic push");
        _Pragma("clang diagnostic ignored \"-Winline-asm\"");
        asm volatile(
            ";-------------------------------------------------------------\n"
            " s_mov_b32 s52, 0x07060302 ; v_perm\n"
            " s_mov_b64 s[38:39], exec ; save current exec\n"
            " s_mov_b32 s8,    %[s_res_o0] \n"
            " s_mov_b32 s9,    %[s_res_o1] \n"
            " s_mov_b32 s12,    %[s_res_b0] \n"
            " s_mov_b32 s13,    %[s_res_b1] \n"
            " s_mov_b32 s14,    %[s_res_b2] \n"
            " s_mov_b32 s15,    %[s_res_b3] \n"
            " ds_read_b64   v[128:129], %[v_sld_y_os] offset:0 + %[sld_a_base]                       \n"
            " ds_read_b64   v[130:131], %[v_sld_y_os] offset:128 + %[sld_a_base]                     \n"
            " ds_read_b64   v[132:133], %[v_sld_y_os] offset:1024 + %[sld_a_base]                    \n"
            " ds_read_b64   v[134:135], %[v_sld_y_os] offset:1152 + %[sld_a_base]                    \n"
            " ds_read_b64   v[136:137], %[v_sld_y_os] offset:2048 + %[sld_a_base]                    \n"
            " ds_read_b64   v[138:139], %[v_sld_y_os] offset:2176 + %[sld_a_base]                    \n"
            " ds_read_b64   v[140:141], %[v_sld_y_os] offset:3072 + %[sld_a_base]                    \n"
            " ds_read_b64   v[142:143], %[v_sld_y_os] offset:3200 + %[sld_a_base]                    \n"
            " ds_read_b64   v[144:145], %[v_sld_y_os] offset:4096 + %[sld_a_base]                    \n"
            " ds_read_b64   v[146:147], %[v_sld_y_os] offset:4224 + %[sld_a_base]                    \n"
            " ds_read_b64   v[148:149], %[v_sld_y_os] offset:5120 + %[sld_a_base]                    \n"
            " ds_read_b64   v[150:151], %[v_sld_y_os] offset:5248 + %[sld_a_base]                    \n"
            " ds_read_b64   v[152:153], %[v_sld_y_os] offset:6144 + %[sld_a_base]                    \n"
            " ds_read_b64   v[154:155], %[v_sld_y_os] offset:6272 + %[sld_a_base]                    \n"
            " ds_read_b64   v[156:157], %[v_sld_y_os] offset:7168 + %[sld_a_base]                    \n"
            " ds_read_b64   v[158:159], %[v_sld_y_os] offset:7296 + %[sld_a_base]                    \n"
            " ds_read_b64   v[160:161], %[v_sld_y_os] offset:8192 + %[sld_a_base]                    \n"
            " ds_read_b64   v[162:163], %[v_sld_y_os] offset:8320 + %[sld_a_base]                    \n"
            " ds_read_b64   v[164:165], %[v_sld_y_os] offset:9216 + %[sld_a_base]                    \n"
            " ds_read_b64   v[166:167], %[v_sld_y_os] offset:9344 + %[sld_a_base]                    \n"
            " ds_read_b64   v[168:169], %[v_sld_y_os] offset:10240 + %[sld_a_base]                    \n"
            " ds_read_b64   v[170:171], %[v_sld_y_os] offset:10368 + %[sld_a_base]                    \n"
            " ds_read_b64   v[172:173], %[v_sld_y_os] offset:11264 + %[sld_a_base]                    \n"
            " ds_read_b64   v[174:175], %[v_sld_y_os] offset:11392 + %[sld_a_base]                    \n"
            " ds_read_b64   v[176:177], %[v_sld_y_os] offset:12288 + %[sld_a_base]                    \n"
            " ds_read_b64   v[178:179], %[v_sld_y_os] offset:12416 + %[sld_a_base]                    \n"
            " ds_read_b64   v[180:181], %[v_sld_y_os] offset:13312 + %[sld_a_base]                    \n"
            " ds_read_b64   v[182:183], %[v_sld_y_os] offset:13440 + %[sld_a_base]                    \n"
            " ds_read_b64   v[184:185], %[v_sld_y_os] offset:14336 + %[sld_a_base]                    \n"
            " ds_read_b64   v[186:187], %[v_sld_y_os] offset:14464 + %[sld_a_base]                    \n"
            " ds_read_b64   v[188:189], %[v_sld_y_os] offset:15360 + %[sld_a_base]                    \n"
            " ds_read_b64   v[190:191], %[v_sld_y_os] offset:15488 + %[sld_a_base]                    \n"
            " ds_read_b64   v[192:193], %[v_sld_y_os] offset:16384 + %[sld_a_base]                    \n"
            " ds_read_b64   v[194:195], %[v_sld_y_os] offset:16512 + %[sld_a_base]                    \n"
            " ds_read_b64   v[196:197], %[v_sld_y_os] offset:17408 + %[sld_a_base]                    \n"
            " ds_read_b64   v[198:199], %[v_sld_y_os] offset:17536 + %[sld_a_base]                    \n"
            " ds_read_b64   v[200:201], %[v_sld_y_os] offset:18432 + %[sld_a_base]                    \n"
            " ds_read_b64   v[202:203], %[v_sld_y_os] offset:18560 + %[sld_a_base]                    \n"
            " ds_read_b64   v[204:205], %[v_sld_y_os] offset:19456 + %[sld_a_base]                    \n"
            " ds_read_b64   v[206:207], %[v_sld_y_os] offset:19584 + %[sld_a_base]                    \n"
            " ds_read_b64   v[208:209], %[v_sld_y_os] offset:20480 + %[sld_a_base]                    \n"
            " ds_read_b64   v[210:211], %[v_sld_y_os] offset:20608 + %[sld_a_base]                    \n"
            " ds_read_b64   v[212:213], %[v_sld_y_os] offset:21504 + %[sld_a_base]                    \n"
            " ds_read_b64   v[214:215], %[v_sld_y_os] offset:21632 + %[sld_a_base]                    \n"
            " ds_read_b64   v[216:217], %[v_sld_y_os] offset:22528 + %[sld_a_base]                    \n"
            " ds_read_b64   v[218:219], %[v_sld_y_os] offset:22656 + %[sld_a_base]                    \n"
            " ds_read_b64   v[220:221], %[v_sld_y_os] offset:23552 + %[sld_a_base]                    \n"
            " ds_read_b64   v[222:223], %[v_sld_y_os] offset:23680 + %[sld_a_base]                    \n"
            " ds_read_b64   v[224:225], %[v_sld_y_os] offset:24576 + %[sld_a_base]                    \n"
            " ds_read_b64   v[226:227], %[v_sld_y_os] offset:24704 + %[sld_a_base]                    \n"
            " ds_read_b64   v[228:229], %[v_sld_y_os] offset:25600 + %[sld_a_base]                    \n"
            " ds_read_b64   v[230:231], %[v_sld_y_os] offset:25728 + %[sld_a_base]                    \n"
            " ds_read_b64   v[232:233], %[v_sld_y_os] offset:26624 + %[sld_a_base]                    \n"
            " ds_read_b64   v[234:235], %[v_sld_y_os] offset:26752 + %[sld_a_base]                    \n"
            " ds_read_b64   v[236:237], %[v_sld_y_os] offset:27648 + %[sld_a_base]                    \n"
            " ds_read_b64   v[238:239], %[v_sld_y_os] offset:27776 + %[sld_a_base]                    \n"
            " ds_read_b64   v[240:241], %[v_sld_y_os] offset:28672 + %[sld_a_base]                    \n"
            " ds_read_b64   v[242:243], %[v_sld_y_os] offset:28800 + %[sld_a_base]                    \n"
            " ds_read_b64   v[244:245], %[v_sld_y_os] offset:29696 + %[sld_a_base]                    \n"
            " ds_read_b64   v[246:247], %[v_sld_y_os] offset:29824 + %[sld_a_base]                    \n"
            " ds_read_b64   v[248:249], %[v_sld_y_os] offset:30720 + %[sld_a_base]                    \n"
            " ds_read_b64   v[250:251], %[v_sld_y_os] offset:30848 + %[sld_a_base]                    \n"
            " ds_read_b64   v[252:253], %[v_sld_y_os] offset:31744 + %[sld_a_base]                    \n"
            " ds_read_b64   v[254:255], %[v_sld_y_os] offset:31872 + %[sld_a_base]                    \n"
            "  s_waitcnt 0                    \n"
            "  buffer_load_dwordx4  acc[0:3], %[v_os_b0], s[12:15], 0 offen \n"
            "  buffer_load_dwordx4  acc[4:7], %[v_os_b0], s[12:15], 0 offen offset:1024 \n"
            "  buffer_load_dwordx4  acc[8:11], %[v_os_b0], s[12:15], 0 offen offset:2048 \n"
            "  buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[12:15], 0 offen offset:3072 \n"
            "  buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[12:15], 0 offen \n"
            "  buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[12:15], 0 offen offset:1024 \n"
            "  buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[12:15], 0 offen offset:2048 \n"
            "  buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[12:15], 0 offen offset:3072 \n"
            "  buffer_load_dwordx4  acc[32:35], %[v_os_b2], s[12:15], 0 offen \n"
            "  buffer_load_dwordx4  acc[36:39], %[v_os_b2], s[12:15], 0 offen offset:1024 \n"
            "  buffer_load_dwordx4  acc[40:43], %[v_os_b2], s[12:15], 0 offen offset:2048 \n"
            "  buffer_load_dwordx4  acc[44:47], %[v_os_b2], s[12:15], 0 offen offset:3072 \n"
            "  buffer_load_dwordx4  acc[48:51], %[v_os_b3], s[12:15], 0 offen \n"
            "  buffer_load_dwordx4  acc[52:55], %[v_os_b3], s[12:15], 0 offen offset:1024 \n"
            "  buffer_load_dwordx4  acc[56:59], %[v_os_b3], s[12:15], 0 offen offset:2048 \n"
            "  buffer_load_dwordx4  acc[60:63], %[v_os_b3], s[12:15], 0 offen offset:3072 \n"
            "  buffer_load_dwordx4  acc[64:67], %[v_os_b4], s[12:15], 0 offen \n"
            "  buffer_load_dwordx4  acc[68:71], %[v_os_b4], s[12:15], 0 offen offset:1024 \n"
            "  buffer_load_dwordx4  acc[72:75], %[v_os_b4], s[12:15], 0 offen offset:2048 \n"
            "  buffer_load_dwordx4  acc[76:79], %[v_os_b4], s[12:15], 0 offen offset:3072 \n"
            "  buffer_load_dwordx4  acc[80:83], %[v_os_b5], s[12:15], 0 offen \n"
            "  buffer_load_dwordx4  acc[84:87], %[v_os_b5], s[12:15], 0 offen offset:1024 \n"
            "  buffer_load_dwordx4  acc[88:91], %[v_os_b5], s[12:15], 0 offen offset:2048 \n"
            "  buffer_load_dwordx4  acc[92:95], %[v_os_b5], s[12:15], 0 offen offset:3072 \n"
            "  buffer_load_dwordx4  acc[96:99], %[v_os_b6], s[12:15], 0 offen \n"
            "  buffer_load_dwordx4  acc[100:103], %[v_os_b6], s[12:15], 0 offen offset:1024 \n"
            "  buffer_load_dwordx4  acc[104:107], %[v_os_b6], s[12:15], 0 offen offset:2048 \n"
            "  buffer_load_dwordx4  acc[108:111], %[v_os_b6], s[12:15], 0 offen offset:3072 \n"
            "  buffer_load_dwordx4  acc[112:115], %[v_os_b7], s[12:15], 0 offen \n"
            "  buffer_load_dwordx4  acc[116:119], %[v_os_b7], s[12:15], 0 offen offset:1024 \n"
            "  buffer_load_dwordx4  acc[120:123], %[v_os_b7], s[12:15], 0 offen offset:2048 \n"
            "  buffer_load_dwordx4  acc[124:127], %[v_os_b7], s[12:15], 0 offen offset:3072 \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
            "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
            "  s_add_u32     s12, s86, s12                                  \n"
            "  s_addc_u32    s13, 0, s13                                    \n"
            "L_start%=:                    \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  s_barrier                                             \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[0:1], v[128:129], 0 \n"
            "  buffer_load_dwordx4  acc[128:131], %[v_os_b0], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[2:3], v[130:131], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[4:5], v[132:133], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[6:7], v[134:135], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[8:9], v[136:137], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  buffer_load_dwordx4  acc[132:135], %[v_os_b0], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[10:11], v[138:139], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[12:13], v[140:141], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[14:15], v[142:143], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[0:1], v[192:193], 0 \n"
            "  buffer_load_dwordx4  acc[136:139], %[v_os_b0], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[2:3], v[194:195], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[4:5], v[196:197], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[6:7], v[198:199], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[8:9], v[200:201], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  buffer_load_dwordx4  acc[140:143], %[v_os_b0], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[10:11], v[202:203], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[12:13], v[204:205], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[14:15], v[206:207], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[16:17], v[128:129], 0 \n"
            "  buffer_load_dwordx4  acc[144:147], %[v_os_b1], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[18:19], v[130:131], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[20:21], v[132:133], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[22:23], v[134:135], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[24:25], v[136:137], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  buffer_load_dwordx4  acc[148:151], %[v_os_b1], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[26:27], v[138:139], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[28:29], v[140:141], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[30:31], v[142:143], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[16:17], v[192:193], 0 \n"
            "  buffer_load_dwordx4  acc[152:155], %[v_os_b1], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[18:19], v[194:195], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[20:21], v[196:197], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[22:23], v[198:199], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[24:25], v[200:201], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  buffer_load_dwordx4  acc[156:159], %[v_os_b1], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[26:27], v[202:203], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[28:29], v[204:205], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[30:31], v[206:207], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[32:33], v[144:145], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  buffer_load_dwordx4  acc[160:163], %[v_os_b2], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[34:35], v[146:147], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[36:37], v[148:149], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[38:39], v[150:151], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[40:41], v[152:153], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  buffer_load_dwordx4  acc[164:167], %[v_os_b2], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[42:43], v[154:155], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[44:45], v[156:157], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[46:47], v[158:159], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[32:33], v[208:209], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  buffer_load_dwordx4  acc[168:171], %[v_os_b2], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[34:35], v[210:211], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[36:37], v[212:213], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[38:39], v[214:215], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[40:41], v[216:217], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  buffer_load_dwordx4  acc[172:175], %[v_os_b2], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[42:43], v[218:219], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[44:45], v[220:221], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[46:47], v[222:223], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[48:49], v[144:145], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  buffer_load_dwordx4  acc[176:179], %[v_os_b3], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[50:51], v[146:147], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[52:53], v[148:149], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[54:55], v[150:151], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[56:57], v[152:153], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  buffer_load_dwordx4  acc[180:183], %[v_os_b3], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[58:59], v[154:155], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[60:61], v[156:157], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[62:63], v[158:159], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[48:49], v[208:209], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  buffer_load_dwordx4  acc[184:187], %[v_os_b3], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[50:51], v[210:211], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[52:53], v[212:213], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[54:55], v[214:215], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[56:57], v[216:217], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  buffer_load_dwordx4  acc[188:191], %[v_os_b3], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[58:59], v[218:219], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[60:61], v[220:221], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[62:63], v[222:223], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[64:65], v[160:161], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  buffer_load_dwordx4  acc[192:195], %[v_os_b4], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[66:67], v[162:163], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[68:69], v[164:165], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[70:71], v[166:167], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[72:73], v[168:169], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  buffer_load_dwordx4  acc[196:199], %[v_os_b4], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[74:75], v[170:171], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[76:77], v[172:173], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[78:79], v[174:175], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[64:65], v[224:225], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  buffer_load_dwordx4  acc[200:203], %[v_os_b4], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[66:67], v[226:227], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[68:69], v[228:229], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[70:71], v[230:231], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[72:73], v[232:233], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  buffer_load_dwordx4  acc[204:207], %[v_os_b4], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[74:75], v[234:235], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[76:77], v[236:237], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[78:79], v[238:239], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[80:81], v[160:161], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  buffer_load_dwordx4  acc[208:211], %[v_os_b5], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[82:83], v[162:163], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[84:85], v[164:165], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[86:87], v[166:167], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[88:89], v[168:169], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  buffer_load_dwordx4  acc[212:215], %[v_os_b5], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[90:91], v[170:171], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[92:93], v[172:173], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[94:95], v[174:175], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[80:81], v[224:225], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  buffer_load_dwordx4  acc[216:219], %[v_os_b5], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[82:83], v[226:227], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[84:85], v[228:229], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[86:87], v[230:231], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[88:89], v[232:233], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  buffer_load_dwordx4  acc[220:223], %[v_os_b5], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[90:91], v[234:235], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[92:93], v[236:237], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[94:95], v[238:239], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[96:97], v[176:177], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  buffer_load_dwordx4  acc[224:227], %[v_os_b6], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[98:99], v[178:179], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[100:101], v[180:181], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[102:103], v[182:183], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[104:105], v[184:185], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  buffer_load_dwordx4  acc[228:231], %[v_os_b6], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[106:107], v[186:187], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[108:109], v[188:189], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c0], %[c1], %[c2], %[c3]], acc[110:111], v[190:191], [%[c0], %[c1], %[c2], %[c3]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[96:97], v[240:241], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  buffer_load_dwordx4  acc[232:235], %[v_os_b6], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[98:99], v[242:243], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[100:101], v[244:245], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[102:103], v[246:247], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[104:105], v[248:249], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  buffer_load_dwordx4  acc[236:239], %[v_os_b6], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[106:107], v[250:251], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[108:109], v[252:253], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c4], %[c5], %[c6], %[c7]], acc[110:111], v[254:255], [%[c4], %[c5], %[c6], %[c7]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[112:113], v[176:177], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  buffer_load_dwordx4  acc[240:243], %[v_os_b7], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[114:115], v[178:179], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[116:117], v[180:181], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[118:119], v[182:183], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[120:121], v[184:185], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  buffer_load_dwordx4  acc[244:247], %[v_os_b7], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[122:123], v[186:187], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[124:125], v[188:189], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c8], %[c9], %[c10], %[c11]], acc[126:127], v[190:191], [%[c8], %[c9], %[c10], %[c11]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[112:113], v[240:241], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  buffer_load_dwordx4  acc[248:251], %[v_os_b7], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[114:115], v[242:243], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[116:117], v[244:245], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[118:119], v[246:247], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[120:121], v[248:249], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  buffer_load_dwordx4  acc[252:255], %[v_os_b7], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[122:123], v[250:251], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[124:125], v[252:253], [%[c12], %[c13], %[c14], %[c15]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c12], %[c13], %[c14], %[c15]], acc[126:127], v[254:255], [%[c12], %[c13], %[c14], %[c15]]\n"
            // "  s_add_u32     s60, 0x00000100, s80                    \n"
            // "  s_cmp_lt_u32  s60, s81                                \n"
            // "  s_cselect_b32  s56, s56, 0                            \n"
            // "  s_add_u32     s12, s56, s12                           \n"
            // "  s_addc_u32    s13, 0, s13                             \n"
            "  v_mul_f32     %[c0], %[scale_0], %[c0]                            \n"
            "  v_mul_f32     %[c1], %[scale_0], %[c1]                            \n"
            "  v_mul_f32     %[c2], %[scale_0], %[c2]                            \n"
            "  v_mul_f32     %[c3], %[scale_0], %[c3]                            \n"
            "  v_mul_f32     %[c4], %[scale_1], %[c4]                            \n"
            "  v_mul_f32     %[c5], %[scale_1], %[c5]                            \n"
            "  v_mul_f32     %[c6], %[scale_1], %[c6]                            \n"
            "  v_mul_f32     %[c7], %[scale_1], %[c7]                            \n"
            "  v_mul_f32     %[c8], %[scale_0], %[c8]                            \n"
            "  v_mul_f32     %[c9], %[scale_0], %[c9]                            \n"
            "  v_mul_f32     %[c10], %[scale_0], %[c10]                            \n"
            "  v_mul_f32     %[c11], %[scale_0], %[c11]                            \n"
            "  v_mul_f32     %[c12], %[scale_1], %[c12]                            \n"
            "  v_mul_f32     %[c13], %[scale_1], %[c13]                            \n"
            "  v_mul_f32     %[c14], %[scale_1], %[c14]                            \n"
            "  v_mul_f32     %[c15], %[scale_1], %[c15]                            \n"
            "  v_cmp_u_f32   s[32:33], %[c0], %[c0]                      \n"
            "  v_add3_u32    v50, %[c0], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c1], %[c1]                      \n"
            "  v_add3_u32    v50, %[c1], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c0], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c2], %[c2]                      \n"
            "  v_add3_u32    v50, %[c2], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c3], %[c3]                      \n"
            "  v_add3_u32    v50, %[c3], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c1], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c4], %[c4]                      \n"
            "  v_add3_u32    v50, %[c4], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c5], %[c5]                      \n"
            "  v_add3_u32    v50, %[c5], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c2], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c6], %[c6]                      \n"
            "  v_add3_u32    v50, %[c6], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c7], %[c7]                      \n"
            "  v_add3_u32    v50, %[c7], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c3], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c8], %[c8]                      \n"
            "  v_add3_u32    v50, %[c8], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c9], %[c9]                      \n"
            "  v_add3_u32    v50, %[c9], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c4], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c10], %[c10]                      \n"
            "  v_add3_u32    v50, %[c10], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c11], %[c11]                      \n"
            "  v_add3_u32    v50, %[c11], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c5], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c12], %[c12]                      \n"
            "  v_add3_u32    v50, %[c12], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c13], %[c13]                      \n"
            "  v_add3_u32    v50, %[c13], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c6], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c14], %[c14]                      \n"
            "  v_add3_u32    v50, %[c14], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c15], %[c15]                      \n"
            "  v_add3_u32    v50, %[c15], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c7], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c0],%[c1]] offset:0    + %[shfl_base]               \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c2],%[c3]] offset:4352 + %[shfl_base]               \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c4],%[c5]] offset:2176 + %[shfl_base]               \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c6],%[c7]] offset:6528 + %[shfl_base]               \n"
            "  s_waitcnt     lgkmcnt(0)                              \n"
            "  s_barrier                                             \n"
            "  ds_read_b32   %[c0], %[v_sfl_sld] offset:0    + %[shfl_base]                    \n"
            "  ds_read_b32   %[c1], %[v_sfl_sld] offset:32   + %[shfl_base]                    \n"
            "  ds_read_b32   %[c2], %[v_sfl_sld] offset:64   + %[shfl_base]                    \n"
            "  ds_read_b32   %[c3], %[v_sfl_sld] offset:96   + %[shfl_base]                    \n"
            "  ds_read_b32   %[c4], %[v_sfl_sld] offset:4352 + %[shfl_base]                    \n"
            "  ds_read_b32   %[c5], %[v_sfl_sld] offset:4384 + %[shfl_base]                    \n"
            "  ds_read_b32   %[c6], %[v_sfl_sld] offset:4416 + %[shfl_base]                    \n"
            "  ds_read_b32   %[c7], %[v_sfl_sld] offset:4448 + %[shfl_base]                    \n"
            "  s_waitcnt     lgkmcnt(0)                              \n"
            //  "s_endpgm\n"
            "  s_mov_b64     exec, %[s_execflag_0]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o0], %[c0], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_1]                    \n"
            
            "  global_atomic_pk_add_bf16   %[v_os_o1], %[c1], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_2]                    \n"

            "  global_atomic_pk_add_bf16   %[v_os_o2], %[c2], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_3]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o3], %[c3], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_4]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o4], %[c4], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_5]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o5], %[c5], s[8:9]  \n"
            // "s_endpgm\n"
            "  s_mov_b64     exec, %[s_execflag_6]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o6], %[c6], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_7]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o7], %[c7], s[8:9]  \n"
            "  s_mov_b64     exec, s[38:39]                           \n"
            "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1     ; k--      \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 0                                \n"
            "  s_cbranch_scc0 L_end%=                                       \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
            "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
            "  s_add_u32     s12, s86, s12                                  \n"
            "  s_addc_u32    s13, 0, s13                                    \n"
            "  s_add_u32     s8, %[s_tile_os_o], s8                             \n"
            "  s_addc_u32    s9, 0, s9                               \n"
                
            //"  s_addk_i32    s80, 0x0080                             \n"
            //"  s_cmp_lt_i32  s80, s81                                \n"
            //"  s_cbranch_scc0  label_0E98                            \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  s_barrier                                             \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[128:129], v[128:129], 0 \n"
            "  buffer_load_dwordx4  acc[0:3], %[v_os_b0], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[130:131], v[130:131], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[132:133], v[132:133], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[134:135], v[134:135], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[136:137], v[136:137], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  buffer_load_dwordx4  acc[4:7], %[v_os_b0], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[138:139], v[138:139], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[140:141], v[140:141], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[142:143], v[142:143], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[128:129], v[192:193], 0 \n"
            "  buffer_load_dwordx4  acc[8:11], %[v_os_b0], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[130:131], v[194:195], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[132:133], v[196:197], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[134:135], v[198:199], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[136:137], v[200:201], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  buffer_load_dwordx4  acc[12:15], %[v_os_b0], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[138:139], v[202:203], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[140:141], v[204:205], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[142:143], v[206:207], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[144:145], v[128:129], 0 \n"
            "  buffer_load_dwordx4  acc[16:19], %[v_os_b1], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[146:147], v[130:131], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[148:149], v[132:133], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[150:151], v[134:135], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[152:153], v[136:137], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  buffer_load_dwordx4  acc[20:23], %[v_os_b1], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[154:155], v[138:139], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[156:157], v[140:141], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[158:159], v[142:143], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[144:145], v[192:193], 0 \n"
            "  buffer_load_dwordx4  acc[24:27], %[v_os_b1], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[146:147], v[194:195], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[148:149], v[196:197], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[150:151], v[198:199], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[152:153], v[200:201], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  buffer_load_dwordx4  acc[28:31], %[v_os_b1], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[154:155], v[202:203], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[156:157], v[204:205], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[158:159], v[206:207], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[160:161], v[144:145], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  buffer_load_dwordx4  acc[32:35], %[v_os_b2], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[162:163], v[146:147], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[164:165], v[148:149], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[166:167], v[150:151], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[168:169], v[152:153], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  buffer_load_dwordx4  acc[36:39], %[v_os_b2], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[170:171], v[154:155], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[172:173], v[156:157], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[174:175], v[158:159], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[160:161], v[208:209], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  buffer_load_dwordx4  acc[40:43], %[v_os_b2], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[162:163], v[210:211], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[164:165], v[212:213], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[166:167], v[214:215], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[168:169], v[216:217], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  buffer_load_dwordx4  acc[44:47], %[v_os_b2], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[170:171], v[218:219], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[172:173], v[220:221], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[174:175], v[222:223], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[176:177], v[144:145], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  buffer_load_dwordx4  acc[48:51], %[v_os_b3], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[178:179], v[146:147], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[180:181], v[148:149], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[182:183], v[150:151], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[184:185], v[152:153], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  buffer_load_dwordx4  acc[52:55], %[v_os_b3], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[186:187], v[154:155], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[188:189], v[156:157], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[190:191], v[158:159], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[176:177], v[208:209], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  buffer_load_dwordx4  acc[56:59], %[v_os_b3], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[178:179], v[210:211], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[180:181], v[212:213], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[182:183], v[214:215], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[184:185], v[216:217], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  buffer_load_dwordx4  acc[60:63], %[v_os_b3], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[186:187], v[218:219], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[188:189], v[220:221], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[190:191], v[222:223], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[192:193], v[160:161], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  buffer_load_dwordx4  acc[64:67], %[v_os_b4], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[194:195], v[162:163], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[196:197], v[164:165], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[198:199], v[166:167], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[200:201], v[168:169], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  buffer_load_dwordx4  acc[68:71], %[v_os_b4], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[202:203], v[170:171], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[204:205], v[172:173], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[206:207], v[174:175], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[192:193], v[224:225], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  buffer_load_dwordx4  acc[72:75], %[v_os_b4], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[194:195], v[226:227], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[196:197], v[228:229], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[198:199], v[230:231], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[200:201], v[232:233], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  buffer_load_dwordx4  acc[76:79], %[v_os_b4], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[202:203], v[234:235], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[204:205], v[236:237], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[206:207], v[238:239], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[208:209], v[160:161], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  buffer_load_dwordx4  acc[80:83], %[v_os_b5], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[210:211], v[162:163], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[212:213], v[164:165], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[214:215], v[166:167], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[216:217], v[168:169], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  buffer_load_dwordx4  acc[84:87], %[v_os_b5], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[218:219], v[170:171], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[220:221], v[172:173], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[222:223], v[174:175], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[208:209], v[224:225], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  buffer_load_dwordx4  acc[88:91], %[v_os_b5], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[210:211], v[226:227], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[212:213], v[228:229], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[214:215], v[230:231], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[216:217], v[232:233], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  buffer_load_dwordx4  acc[92:95], %[v_os_b5], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[218:219], v[234:235], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[220:221], v[236:237], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[222:223], v[238:239], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  s_waitcnt     vmcnt(32)                               \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[224:225], v[176:177], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  buffer_load_dwordx4  acc[96:99], %[v_os_b6], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[226:227], v[178:179], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[228:229], v[180:181], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[230:231], v[182:183], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[232:233], v[184:185], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  buffer_load_dwordx4  acc[100:103], %[v_os_b6], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[234:235], v[186:187], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[236:237], v[188:189], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c16],%[c17],%[c18],%[c19]], acc[238:239], v[190:191], [%[c16],%[c17],%[c18],%[c19]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[224:225], v[240:241], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  buffer_load_dwordx4  acc[104:107], %[v_os_b6], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[226:227], v[242:243], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[228:229], v[244:245], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[230:231], v[246:247], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[232:233], v[248:249], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  buffer_load_dwordx4  acc[108:111], %[v_os_b6], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[234:235], v[250:251], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[236:237], v[252:253], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c20],%[c21],%[c22],%[c23]], acc[238:239], v[254:255], [%[c20],%[c21],%[c22],%[c23]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[240:241], v[176:177], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  buffer_load_dwordx4  acc[112:115], %[v_os_b7], s[12:15], 0 offen \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[242:243], v[178:179], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[244:245], v[180:181], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[246:247], v[182:183], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[248:249], v[184:185], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  buffer_load_dwordx4  acc[116:119], %[v_os_b7], s[12:15], 0 offen offset:1024 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[250:251], v[186:187], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[252:253], v[188:189], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c24],%[c25],%[c26],%[c27]], acc[254:255], v[190:191], [%[c24],%[c25],%[c26],%[c27]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[240:241], v[240:241], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  buffer_load_dwordx4  acc[120:123], %[v_os_b7], s[12:15], 0 offen offset:2048 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[242:243], v[242:243], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[244:245], v[244:245], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[246:247], v[246:247], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[248:249], v[248:249], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  buffer_load_dwordx4  acc[124:127], %[v_os_b7], s[12:15], 0 offen offset:3072 \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[250:251], v[250:251], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[252:253], v[252:253], [%[c28],%[c29],%[c30],%[c31]] \n"
            "  v_mfma_f32_16x16x16_bf16  [%[c28],%[c29],%[c30],%[c31]], acc[254:255], v[254:255], [%[c28],%[c29],%[c30],%[c31]]\n"
            // "  s_add_u32     s60, 0x00000100, s80                    \n"
            // "  s_cmp_lt_u32  s60, s81                                \n"
            // "  s_cselect_b32  s56, s56, 0                            \n"
            // "  s_add_u32     s12, s56, s12                           \n"
            // "  s_addc_u32    s13, 0, s13                             \n"
            "  v_mul_f32     %[c16], %[scale_0], %[c16]                            \n"
            "  v_mul_f32     %[c17], %[scale_0], %[c17]                            \n"
            "  v_mul_f32     %[c18], %[scale_0], %[c18]                            \n"
            "  v_mul_f32     %[c19], %[scale_0], %[c19]                            \n"
            "  v_mul_f32     %[c20], %[scale_1], %[c20]                            \n"
            "  v_mul_f32     %[c21], %[scale_1], %[c21]                            \n"
            "  v_mul_f32     %[c22], %[scale_1], %[c22]                            \n"
            "  v_mul_f32     %[c23], %[scale_1], %[c23]                            \n"
            "  v_mul_f32     %[c24], %[scale_0], %[c24]                            \n"
            "  v_mul_f32     %[c25], %[scale_0], %[c25]                            \n"
            "  v_mul_f32     %[c26], %[scale_0], %[c26]                            \n"
            "  v_mul_f32     %[c27], %[scale_0], %[c27]                            \n"
            "  v_mul_f32     %[c28], %[scale_1], %[c28]                            \n"
            "  v_mul_f32     %[c29], %[scale_1], %[c29]                            \n"
            "  v_mul_f32     %[c30], %[scale_1], %[c30]                            \n"
            "  v_mul_f32     %[c31], %[scale_1], %[c31]                            \n"
            "  v_cmp_u_f32   s[32:33], %[c16], %[c16]                      \n"
            "  v_add3_u32    v50, %[c16], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c17], %[c17]                      \n"
            "  v_add3_u32    v50, %[c17], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c16], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c18], %[c18]                      \n"
            "  v_add3_u32    v50, %[c18], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c19], %[c19]                      \n"
            "  v_add3_u32    v50, %[c19], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c17], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c20], %[c20]                      \n"
            "  v_add3_u32    v50, %[c20], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c21], %[c21]                      \n"
            "  v_add3_u32    v50, %[c21], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c18], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c22], %[c22]                      \n"
            "  v_add3_u32    v50, %[c22], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c23], %[c23]                      \n"
            "  v_add3_u32    v50, %[c23], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c19], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c24], %[c24]                      \n"
            "  v_add3_u32    v50, %[c24], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c25], %[c25]                      \n"
            "  v_add3_u32    v50, %[c25], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c20], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c26], %[c26]                      \n"
            "  v_add3_u32    v50, %[c26], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c27], %[c27]                      \n"
            "  v_add3_u32    v50, %[c27], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c21], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c28], %[c28]                      \n"
            "  v_add3_u32    v50, %[c28], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c29], %[c29]                      \n"
            "  v_add3_u32    v50, %[c29], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c22], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  v_cmp_u_f32   s[32:33], %[c30], %[c30]                      \n"
            "  v_add3_u32    v50, %[c30], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v54, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_cmp_u_f32   s[32:33], %[c31], %[c31]                      \n"
            "  v_add3_u32    v50, %[c31], %[v_nan_lo], 1                        \n"
            "  v_cndmask_b32  v55, v50, %[v_nan_hi], s[32:33]                \n"
            "  v_perm_b32    %[c23], v55, v54, s52                      \n"
            "  ;------------------------------  \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c16],%[c17]] offset:0    + %[shfl_base]         \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c18],%[c19]] offset:4352 + %[shfl_base]         \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c20],%[c21]] offset:2176 + %[shfl_base]         \n"
            "  ds_write_b64  %[v_sfl_sst], [%[c22],%[c23]] offset:6528 + %[shfl_base]         \n"
            "  s_waitcnt     lgkmcnt(0)                              \n"
            "  s_barrier                                             \n"
            "  ds_read_b32   %[c16], %[v_sfl_sld] offset:0    + %[shfl_base]                  \n"
            "  ds_read_b32   %[c17], %[v_sfl_sld] offset:32   + %[shfl_base]                  \n"
            "  ds_read_b32   %[c18], %[v_sfl_sld] offset:64   + %[shfl_base]                  \n"
            "  ds_read_b32   %[c19], %[v_sfl_sld] offset:96   + %[shfl_base]                  \n"
            "  ds_read_b32   %[c20], %[v_sfl_sld] offset:4352 + %[shfl_base]                  \n"
            "  ds_read_b32   %[c21], %[v_sfl_sld] offset:4384 + %[shfl_base]                  \n"
            "  ds_read_b32   %[c22], %[v_sfl_sld] offset:4416 + %[shfl_base]                  \n"
            "  ds_read_b32   %[c23], %[v_sfl_sld] offset:4448 + %[shfl_base]                  \n"
            "  s_waitcnt     lgkmcnt(0)                              \n"
            "  s_mov_b64     exec, %[s_execflag_0]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o0], %[c0], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_1]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o1], %[c1], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_2]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o2], %[c2], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_3]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o3], %[c3], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_4]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o4], %[c4], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_5]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o5], %[c5], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_6]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o6], %[c6], s[8:9]  \n"
            "  s_mov_b64     exec, %[s_execflag_7]                    \n"
            "  global_atomic_pk_add_bf16   %[v_os_o7], %[c7], s[8:9]  \n"
            "  s_mov_b64     exec, s[38:39]                           \n"
            "  s_sub_i32     %[s_loop_cnt], %[s_loop_cnt], 1     ; k--      \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 0                                \n"
            "  s_cbranch_scc0 L_end%=                                       \n"
            "  s_cmp_gt_i32  %[s_loop_cnt] 1             ; move b with cond \n"
            "  s_cselect_b32 s86, %[s_tile_os_b], 0                          \n"
            "  s_add_u32     s12, s86, s12                                  \n"
            "  s_addc_u32    s13, 0, s13                                    \n"
            "  s_add_u32     s8, %[s_tile_os_o], s8                             \n"
            "  s_addc_u32    s9, 0, s9                               \n"
            "  s_branch      L_start%=          \n"
            "L_end%=:                                                \n"
            :[smem_]"+r"(smem),
            [s_loop_cnt]"+s"(loop_cnt),
                [c0]"+v" (v_c0),
                [c1]"+v" (v_c1),
                [c2]"+v" (v_c2),
                [c3]"+v" (v_c3),
                [c4]"+v" (v_c4),
                [c5]"+v" (v_c5),
                [c6]"+v" (v_c6),
                [c7]"+v" (v_c7),
                [c8]"+v" (v_c8),
                [c9]"+v" (v_c9),
                [c10]"+v"(v_c10),
                [c11]"+v"(v_c11),
                [c12]"+v"(v_c12),
                [c13]"+v"(v_c13),
                [c14]"+v"(v_c14),
                [c15]"+v"(v_c15),
                [c16]"+v"(v_c16),
                [c17]"+v"(v_c17),
                [c18]"+v"(v_c18),
                [c19]"+v"(v_c19),
                [c20]"+v"(v_c20),
                [c21]"+v"(v_c21),
                [c22]"+v"(v_c22),
                [c23]"+v"(v_c23),
                [c24]"+v"(v_c24),
                [c25]"+v"(v_c25),
                [c26]"+v"(v_c26),
                [c27]"+v"(v_c27),
                [c28]"+v"(v_c28),
                [c29]"+v"(v_c29),
                [c30]"+v"(v_c30),
                [c31]"+v"(v_c31)
            :
            [sld_a_base]"n"(0),
            [shfl_base]"n"(0),
            [v_sld_y_os]"v"(sld_y_os),
            [v_sfl_sld]"v"(sfl_sld),
            [v_sfl_sst]"v"(sfl_sst),
            [s_res_o0]"s"(res_o[0]),
                [s_res_o1]"s"(res_o[1]),
                //[s_res_o2]"s"(res_o[2]),
                //[s_res_o3]"s"(res_o[3]),
                [s_res_b0]"s"(res_b[0]),
                [s_res_b1]"s"(res_b[1]),
                [s_res_b2]"s"(res_b[2]),
                [s_res_b3]"s"(res_b[3]),
                [v_os_o0]"v"(static_cast<index_t>(cached_coords_o[number<0>{}] * sizeof(ODataType))),
                [v_os_o1]"v"(static_cast<index_t>(cached_coords_o[number<1>{}] * sizeof(ODataType))),
                [v_os_o2]"v"(static_cast<index_t>(cached_coords_o[number<2>{}] * sizeof(ODataType))),
                [v_os_o3]"v"(static_cast<index_t>(cached_coords_o[number<3>{}] * sizeof(ODataType))),
                [v_os_o4]"v"(static_cast<index_t>(cached_coords_o[number<4>{}] * sizeof(ODataType))),
                [v_os_o5]"v"(static_cast<index_t>(cached_coords_o[number<5>{}] * sizeof(ODataType))),
                [v_os_o6]"v"(static_cast<index_t>(cached_coords_o[number<6>{}] * sizeof(ODataType))),
                [v_os_o7]"v"(static_cast<index_t>(cached_coords_o[number<7>{}] * sizeof(ODataType))),
                [v_os_b0]"v"(static_cast<index_t>(cached_coords_b[number<0>{}] * sizeof(BDataType))),
                [v_os_b1]"v"(static_cast<index_t>(cached_coords_b[number<1>{}] * sizeof(BDataType))),
                [v_os_b2]"v"(static_cast<index_t>(cached_coords_b[number<2>{}] * sizeof(BDataType))),
                [v_os_b3]"v"(static_cast<index_t>(cached_coords_b[number<3>{}] * sizeof(BDataType))),
                [v_os_b4]"v"(static_cast<index_t>(cached_coords_b[number<4>{}] * sizeof(BDataType))),
                [v_os_b5]"v"(static_cast<index_t>(cached_coords_b[number<5>{}] * sizeof(BDataType))),
                [v_os_b6]"v"(static_cast<index_t>(cached_coords_b[number<6>{}] * sizeof(BDataType))),
                [v_os_b7]"v"(static_cast<index_t>(cached_coords_b[number<7>{}] * sizeof(BDataType))),

                [s_tile_os_o]"s"(tile_stride_o_bytes),
                [s_tile_os_b]"s"(tile_stride_b_bytes),
                [scale_0]"v"(s0),
                [scale_1]"v"(s1),
                [v_nan_lo]"v"(nan_lo),
                [v_nan_hi]"v"(nan_hi),
                [s_execflag_0]"s"(o_flags[number<0>{}]),
                [s_execflag_1]"s"(o_flags[number<1>{}]),
                [s_execflag_2]"s"(o_flags[number<2>{}]),
                [s_execflag_3]"s"(o_flags[number<3>{}]),
                [s_execflag_4]"s"(o_flags[number<4>{}]),
                [s_execflag_5]"s"(o_flags[number<5>{}]),
                [s_execflag_6]"s"(o_flags[number<6>{}]),
                [s_execflag_7]"s"(o_flags[number<7>{}])
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
          "s8", "s9", "s12", "s13", "s14", "s15", "s38", "s39", "s52", "s86",
          // "s32", "s33",
          "v50", "v54", "v55",
          "v64","v65","v66","v67","v68","v69","v70","v71",
          "v72","v73","v74","v75","v76","v77","v78","v79",
          "v80","v81","v82","v83","v84","v85","v86","v87",
          "v88","v89","v90","v91","v92","v93","v94","v95",
          "v128", "v129", "v130", "v131",
          "v132", "v133", "v134", "v135", "v136", "v137", "v138", "v139",
          "v140", "v141", "v142", "v143", "v144", "v145", "v146", "v147",
          "v148", "v149", "v150", "v151", "v152", "v153", "v154", "v155",
          "v156", "v157", "v158", "v159", "v160", "v161", "v162", "v163",
          "v164", "v165", "v166", "v167", "v168", "v169", "v170", "v171",
          "v172", "v173", "v174", "v175", "v176", "v177", "v178", "v179",
          "v180", "v181", "v182", "v183", "v184", "v185", "v186", "v187",
          "v188", "v189", "v190", "v191", "v192", "v193", "v194", "v195",
          "v196", "v197", "v198", "v199", "v200", "v201", "v202", "v203",
          "v204", "v205", "v206", "v207", "v208", "v209", "v210", "v211",
          "v212", "v213", "v214", "v215", "v216", "v217", "v218", "v219",
          "v220", "v221", "v222", "v223", "v224", "v225", "v226", "v227",
          "v228", "v229", "v230", "v231", "v232", "v233", "v234", "v235",
          "v236", "v237", "v238", "v239", "v240", "v241", "v242", "v243",
          "v244", "v245", "v246", "v247", "v248", "v249", "v250", "v251",
          "v252", "v253", "v254", "v255"
        );
        _Pragma("clang diagnostic pop");
        // clang-format on
    }
};

} // namespace ck_tile
