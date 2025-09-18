// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>
#include <tuple>
#include <stdexcept>
#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm_quant.hpp"

// Forward declarations for quant type-specific implementations
template <ck_tile::QuantType QT>
struct QuantTypeTraits;

// Base class for common quant gemm functionality
template <typename Tuple, typename Derived>
class TestCkTileGemmQuantBase : public ::testing::Test
{
    protected:
    using ALayout                      = std::tuple_element_t<0, Tuple>;
    using BLayout                      = std::tuple_element_t<1, Tuple>;
    using CLayout                      = std::tuple_element_t<2, Tuple>;
    using ADataType                    = std::tuple_element_t<3, Tuple>;
    using BDataType                    = std::tuple_element_t<4, Tuple>;
    using AccDataType                  = std::tuple_element_t<5, Tuple>;
    using CDataType                    = std::tuple_element_t<6, Tuple>;
    static constexpr auto QuantType    = std::tuple_element_t<7, Tuple>::value;
    using GemmConfig   = std::tuple_element_t<8, Tuple>;
    static constexpr uint32_t QuantGroupSize = std::tuple_element_t<9, Tuple>::value;

    // Get the quant-type specific data types from traits
    using QuantTraits = QuantTypeTraits<QuantType>;
    using AQDataType  = typename QuantTraits::template AQDataType<ADataType, BDataType>;
    using BQDataType  = typename QuantTraits::template BQDataType<ADataType, BDataType>;
    using ComputeDataType = typename QuantTraits::template ComputeDataType<ADataType, BDataType>;

    static constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
    static constexpr ck_tile::index_t N_Tile = GemmConfig::N_Tile;
    static constexpr ck_tile::index_t K_Tile = GemmConfig::K_Tile;

    static constexpr ck_tile::index_t M_Warp = GemmConfig::M_Warp;
    static constexpr ck_tile::index_t N_Warp = GemmConfig::N_Warp;
    static constexpr ck_tile::index_t K_Warp = GemmConfig::K_Warp;

    static constexpr ck_tile::index_t M_Warp_Tile = GemmConfig::M_Warp_Tile;
    static constexpr ck_tile::index_t N_Warp_Tile = GemmConfig::N_Warp_Tile;
    static constexpr ck_tile::index_t K_Warp_Tile = GemmConfig::K_Warp_Tile;

    public:
    void SetUp() override
    {
        static_cast<Derived*>(this)->SetUpQuantTypeSpecific();
    }

    void TearDown() override
    {
        static_cast<Derived*>(this)->TearDownQuantTypeSpecific();
    }

    // Common test execution logic
    template <bool PadM, bool PadN, bool PadK, bool Preshuffle>
    void invoke_quant_gemm(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& s)
    {
        constexpr bool kPadM      = PadM;
        constexpr bool kPadN      = PadN;
        constexpr bool kPadK      = PadK;
        constexpr bool preshuffle = Preshuffle;

        constexpr int kOccupancy [[maybe_unused]] = 1;

        using CodegenGemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

        using CodegenGemmTraits = ck_tile::TileGemmQuantTraits<kPadM,
                                                               kPadN,
                                                               kPadK,
                                                               preshuffle,
                                                               ALayout,
                                                               BLayout,
                                                               CLayout,
                                                               QuantType>;

        // Let the derived class create the appropriate pipeline and epilogue
        static_cast<Derived*>(this)->template run_quant_gemm_impl<CodegenGemmShape, 
                                                                   TilePartitioner,
                                                                   CodegenGemmTraits>(args, s);
    }

    void RunTest(ck_tile::index_t M, ck_tile::index_t N, ck_tile::index_t K)
    {
        auto args = static_cast<Derived*>(this)->generate_test_data(M, N, K);
        ck_tile::stream_config stream_config{};
        
        // Test different combinations of padding and preshuffle
        invoke_quant_gemm<false, false, false, false>(args, stream_config);
    }

    // Helper function to check layout 
    template<typename Layout>
    static constexpr auto is_row_major(Layout)
    {
        return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(Layout{})>, ck_tile::tensor_layout::gemm::RowMajor>>{};
    }
};

// Define generic QuantTypeTraits template (will be specialized)
template <ck_tile::QuantType QT>
struct QuantTypeTraits
{
    static_assert(QT == ck_tile::QuantType::AQuantGrouped ||
                  QT == ck_tile::QuantType::BQuantGrouped ||
                  QT == ck_tile::QuantType::RowColQuant ||
                  QT == ck_tile::QuantType::TensorQuant,
                  "Unsupported quantization type");
};

// Specialization for AQuantGrouped
template <>
struct QuantTypeTraits<ck_tile::QuantType::AQuantGrouped>
{
    template <typename ADataType, typename BDataType>
    using AQDataType = float; // Scale type for A quantization
    
    template <typename ADataType, typename BDataType>
    using BQDataType = BDataType; // No B quantization for AQuant
    
    template <typename ADataType, typename BDataType>
    using ComputeDataType = BDataType; // For AQuant, compute type is BDataType

    static constexpr const char* name = "aquant";
};

// Specialization for BQuantGrouped
template <>
struct QuantTypeTraits<ck_tile::QuantType::BQuantGrouped>
{
    template <typename ADataType, typename BDataType>
    using AQDataType = ADataType; // No A quantization for BQuant
    
    template <typename ADataType, typename BDataType>
    using BQDataType = float; // Scale type for B quantization
    
    template <typename ADataType, typename BDataType>
    using ComputeDataType = ADataType; // For BQuant, compute type is ADataType

    static constexpr const char* name = "bquant";
};

// Specialization for RowColQuant
template <>
struct QuantTypeTraits<ck_tile::QuantType::RowColQuant>
{
    template <typename ADataType, typename BDataType>
    using AQDataType = float; // Scale type for A row/col quantization
    
    template <typename ADataType, typename BDataType>
    using BQDataType = float; // Scale type for B row/col quantization
    
    template <typename ADataType, typename BDataType>
    using ComputeDataType = BDataType; // For RowColQuant, compute type is BDataType

    static constexpr const char* name = "rowcol";
};

// Specialization for TensorQuant
template <>
struct QuantTypeTraits<ck_tile::QuantType::TensorQuant>
{
    template <typename ADataType, typename BDataType>
    using AQDataType = float; // Scale type for A tensor quantization
    
    template <typename ADataType, typename BDataType>
    using BQDataType = float; // Scale type for B tensor quantization
    
    template <typename ADataType, typename BDataType>
    using ComputeDataType = ADataType; // For TensorQuant, compute type is ADataType

    static constexpr const char* name = "tensor";
};
