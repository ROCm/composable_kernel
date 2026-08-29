// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"
#include "test_mx_gemm_pipeline_util.hpp"

template <typename Tuple, typename Derived>
class TestCkTileMxGemmPipelineWmmaBase : public TestCkTileMxGemmPipeline<Tuple, Derived>
{
    public:
    static constexpr bool check_data_type()
    {
        using Base = TestCkTileMxGemmPipeline<Tuple, Derived>;

        if constexpr(!is_valid_mx_scale_combination<typename Base::ADataType,
                                                    typename Base::AScaleDataType,
                                                    typename Base::BDataType,
                                                    typename Base::BScaleDataType>())
        {
            return false;
        }

#if defined(CK_USE_GFX1250)
        using DeviceIp = ck_tile::gfx125_t;
#else
#error "Unsupported architecture for WMMA MX GEMM"
#endif

        return ck_tile::has_wmma_traits_v<DeviceIp,
                                          typename Base::ADataType,
                                          typename Base::BDataType,
                                          typename Base::AccDataType,
                                          ck_tile::constant<Base::M_Warp_Tile>::value,
                                          ck_tile::constant<Base::N_Warp_Tile>::value,
                                          ck_tile::constant<Base::K_Warp_Tile>::value>;
    }

    private:
    template <typename ADataType,
              typename AScaleDataType,
              typename BDataType,
              typename BScaleDataType>
    static constexpr bool is_valid_mx_scale_combination()
    {
        constexpr bool a_is_f4      = std::is_same_v<ADataType, ck_tile::pk_fp4_t>;
        constexpr bool b_is_f4      = std::is_same_v<BDataType, ck_tile::pk_fp4_t>;
        constexpr bool a_scale_e8m0 = std::is_same_v<AScaleDataType, ck_tile::e8m0_t>;
        constexpr bool b_scale_e8m0 = std::is_same_v<BScaleDataType, ck_tile::e8m0_t>;

        // Non-F4 must use E8M0 scale
        if constexpr(!a_is_f4 && !a_scale_e8m0)
            return false;
        if constexpr(!b_is_f4 && !b_scale_e8m0)
            return false;

        // Both E8M0 -> always valid
        if constexpr(a_scale_e8m0 && b_scale_e8m0)
            return true;

        // Both non-E8M0 -> must match (both are F4 by rule 1)
        if constexpr(!a_scale_e8m0 && !b_scale_e8m0)
            return std::is_same_v<AScaleDataType, BScaleDataType>;

        // One side non-E8M0: the E8M0 side must not be F4
        if constexpr(!a_scale_e8m0)
            return !b_is_f4;
        if constexpr(!b_scale_e8m0)
            return !a_is_f4;

        return true;
    }
};
