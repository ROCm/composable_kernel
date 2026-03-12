// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

inline auto& get_kernel_lut()
{
    // In an inline function, function-local static objects in all function definitions are shared
    // across all translation units.
    static std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>> lut;
    return lut;
}

inline size_t hash_multiple_strings(const std::vector<std::string>& inputs)
{
    std::hash<std::string> hasher;
    size_t combined_hash = 0;
    for(const auto& str : inputs)
    {
        // Hash combine using golden ratio constant and bit shifts for good distribution and
        // order-dependent mixing
        combined_hash ^= hasher(str) + 0x9e3779b9 + (combined_hash << 6) + (combined_hash >> 2);
    }
    return combined_hash;
}

template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
{
    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{};
}

template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
                         const ck_tile::index_t kbatch,
                         const float max_accumulated_value)
{
    using ComputeType = std::conditional_t<
        std::is_same_v<BDataType, ck_tile::pk_fp4_t>,
        ADataType,
        std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>>;
    // Calculate thresholds
    const auto rtol = ck_tile::get_relative_threshold<ComputeType, CDataType, AccDataType>(
        ck_tile::integer_divide_ceil(K, kbatch));
    const auto atol = ck_tile::get_absolute_threshold<ComputeType, CDataType, AccDataType>(
        max_accumulated_value / kbatch, ck_tile::integer_divide_ceil(K, kbatch));
    // Calculate error due to split_k accumulation
    const auto rtol_split_k =
        ck_tile::get_relative_threshold<CDataType, CDataType, CDataType>(kbatch);
    const auto atol_split_k = ck_tile::get_absolute_threshold<CDataType, CDataType, CDataType>(
        max_accumulated_value, kbatch);
    // Use higher threshold
    return ck_tile::make_tuple(std::max(rtol, rtol_split_k), std::max(atol, atol_split_k));
}

struct GemmConfigBase
{
    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = true;

    static constexpr bool PermuteA = false;
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu = 1;
    static constexpr auto Scheduler  = ck_tile::GemmPipelineScheduler::Intrawave;

    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;

    static constexpr bool APreshuffleQuant = false;
    static constexpr bool BPreshuffleQuant = false;
    static constexpr bool PreshuffleB      = false;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr bool TiledMMAPermuteN = false;
};

template <typename PrecType>
struct GemmConfigQuantDecode : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    // static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Interwave;
};

template <typename PrecType>
struct GemmConfigQuantDecodeInterwave : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Interwave;
};

template <typename PrecType>
struct GemmConfigRowColQuant : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
};

template <typename PrecType>
struct GemmConfigPreshuffleQuantDecode : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();

    static constexpr bool APreshuffleQuant = true;
    static constexpr bool BPreshuffleQuant = true;
};

template <typename PrecType>
struct GemmConfigPreshuffleB_BQuant_Decode : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 64;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();

    static constexpr bool PreshuffleB      = true;
    static constexpr bool DoubleSmemBuffer = true;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = N_Repeat % 2 == 0;
};

template <typename PrecType>
struct GemmConfigPreshuffleB_PreshuffleBQuant_Decode
    : public GemmConfigPreshuffleB_BQuant_Decode<PrecType>
{
    static constexpr bool BPreshuffleQuant = true;
};

template <typename PrecType>
struct GemmConfigPreshuffleB_BQuant_Prefill : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();

    static constexpr bool PreshuffleB      = true;
    static constexpr bool DoubleSmemBuffer = true;

    static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
    static constexpr bool TiledMMAPermuteN = N_Repeat % 2 == 0;
    static constexpr int kBlockPerCu       = 2;
};

template <typename PrecType>
struct GemmConfigPreshuffleB_PreshuffleBQuant_Prefill
    : public GemmConfigPreshuffleB_BQuant_Prefill<PrecType>
{
    static constexpr bool BPreshuffleQuant = true;
};

template <typename PrecType, bool TransposeC_ = true>
struct GemmConfigPreshuffleB_ABQuant_Prefill : public GemmConfigPreshuffleB_BQuant_Prefill<PrecType>
{
    static constexpr ck_tile::index_t M_Warp = 2;
    static constexpr ck_tile::index_t N_Warp = 2;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr bool kPadK      = false;
    static constexpr bool TransposeC = TransposeC_;
};

template <typename PrecType, bool TransposeC_ = true>
struct GemmConfigPreshuffleB_ABQuant_PreshuffleBQuant_Prefill
    : public GemmConfigPreshuffleB_ABQuant_Prefill<PrecType, TransposeC_>
{
    static constexpr bool BPreshuffleQuant = true;
};

template <typename PrecType, bool TransposeC_ = true>
struct GemmConfigPreshuffleB_ABQuant_Decode : public GemmConfigPreshuffleB_BQuant_Prefill<PrecType>
{
    static constexpr ck_tile::index_t M_Tile = 16;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);

    static constexpr bool kPadK      = false;
    static constexpr bool TransposeC = TransposeC_;
};

template <typename PrecType>
struct GemmConfigQuantPrefill : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile =
        ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();

    // static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Interwave;
};

template <typename PrecType, bool TransposeC_ = true>
struct GemmConfigABQuantPrefill : public GemmConfigQuantPrefill<PrecType>
{
    static constexpr bool kPadK      = false;
    static constexpr bool TransposeC = TransposeC_;
};

// Used for A=16bit and B=8bit. The warp tile has KPack=16
// Matrix A: Vectorsize =  8, KPack=16 -> LDS read/write vectorsize =  8 (128bit)
// Matrix B: Vectorsize = 16, KPack=16 -> LDS read/write vectorsize = 16 (128bit)
struct GemmConfigMixedPrecision : public GemmConfigBase
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128;

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 64;
};

template <typename PrecType, bool TransposeC_ = true>
struct GemmConfigEightWarps : public GemmConfigABQuantPrefill<PrecType, TransposeC_>
{
    static constexpr ck_tile::index_t M_Warp = 4;
    static constexpr ck_tile::index_t N_Warp = 2; // NWarps == 2 for ping-pong!
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Tile = 192;
    static constexpr ck_tile::index_t N_Tile = 128 * N_Warp;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(PrecType) * K_Warp;

    static constexpr bool kPadK      = false;
    static constexpr int kBlockPerCu = 1;
};

template <typename PrecType, bool TransposeC_ = true>
struct GemmConfigPreshuffleBEightWarps : public GemmConfigEightWarps<PrecType, TransposeC_>
{
    static constexpr bool PreshuffleB      = true;
    static constexpr bool DoubleSmemBuffer = true;
};

template <typename PrecType>
struct GemmConfigPreshuffleBQuantPrefill : public GemmConfigQuantPrefill<PrecType>
{
    static constexpr bool BPreshuffleQuant = true;
};

template <typename PrecType>
struct GemmConfigBQuantPrefill_Wmma : public GemmConfigQuantPrefill<PrecType>
{
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

template <typename PrecType>
struct GemmConfigPreshuffleB_BQuant_Prefill_Wmma
    : public GemmConfigPreshuffleB_BQuant_Prefill<PrecType>
{
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

template <typename PrecType>
struct GemmConfigPreshuffleB_PreshuffleBQuant_Prefill_Wmma
    : public GemmConfigPreshuffleB_PreshuffleBQuant_Prefill<PrecType>
{
    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = 16;
};

template <typename ADataType_,
          typename BDataType_ = ADataType_,
          typename CDataType_ = ADataType_,
          typename QDataType_ = float>
struct GemmQuantTypeConfig
{
    using ADataType   = ADataType_;
    using QDataType   = QDataType_;
    using BDataType   = BDataType_;
    using AccDataType = float;
    using CDataType   = CDataType_;
};
