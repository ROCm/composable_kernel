
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"

template <typename DataType>
struct GemmBasicTypeConfig;

template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

using Types = GemmBasicTypeConfig<ck_tile::half_t>;

// Specific type aliases for easy access
using ADataType   = Types::ADataType;
using BDataType   = Types::BDataType;
using AccDataType = Types::AccDataType;
using CDataType   = Types::CDataType;

using Row = ck_tile::tensor_layout::gemm::RowMajor;
using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

struct gemm_traits
{
    std::string data_type;
    bool is_a_rowmajor;
    bool is_b_rowmajor;
    bool is_c_rowmajor;
};

template <typename ADataType_,
          typename BDataType_,
          typename AccDataType_,
          typename CDataType_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          ck_tile::index_t M_Tile_,
          ck_tile::index_t N_Tile_,
          ck_tile::index_t K_Tile_,
          ck_tile::index_t M_Warp_,
          ck_tile::index_t N_Warp_,
          ck_tile::index_t K_Warp_,
          ck_tile::index_t M_Warp_Tile_,
          ck_tile::index_t N_Warp_Tile_,
          ck_tile::index_t K_Warp_Tile_,
          bool kPadM_,
          bool kPadN_,
          bool kPadK_>
struct gemm_traits_
{
    using ADataType                               = ck_tile::remove_cvref_t<ADataType_>;
    using BDataType                               = ck_tile::remove_cvref_t<BDataType_>;
    using AccDataType                             = ck_tile::remove_cvref_t<AccDataType_>;
    using CDataType                               = ck_tile::remove_cvref_t<CDataType_>;
    using ALayout                                 = ck_tile::remove_cvref_t<ALayout_>;
    using BLayout                                 = ck_tile::remove_cvref_t<BLayout_>;
    using CLayout                                 = ck_tile::remove_cvref_t<CLayout_>;
    static constexpr ck_tile::index_t M_Tile      = M_Tile_;
    static constexpr ck_tile::index_t N_Tile      = N_Tile_;
    static constexpr ck_tile::index_t K_Tile      = K_Tile_;
    static constexpr ck_tile::index_t M_Warp      = M_Warp_;
    static constexpr ck_tile::index_t N_Warp      = N_Warp_;
    static constexpr ck_tile::index_t K_Warp      = K_Warp_;
    static constexpr ck_tile::index_t M_Warp_Tile = M_Warp_Tile_;
    static constexpr ck_tile::index_t N_Warp_Tile = N_Warp_Tile_;
    static constexpr ck_tile::index_t K_Warp_Tile = K_Warp_Tile_;
    static constexpr bool kPadM                   = kPadM_;
    static constexpr bool kPadN                   = kPadN_;
    static constexpr bool kPadK                   = kPadK_;
};

// host API
template <typename Traits_>
float gemm_(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s);

float gemm(const gemm_traits& traits,
           const ck_tile::GemmHostArgs& args,
           const ck_tile::stream_config& s);
