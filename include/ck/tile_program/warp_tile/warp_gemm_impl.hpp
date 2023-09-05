// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"

namespace ck {
namespace tile_program {
namespace warp {

template <typename WarpGemmAttribute_>
struct WarpGemmImpl
{
    using WarpGemmAttribute = remove_cvref_t<WarpGemmAttribute_>;

    static constexpr index_t kM = WarpGemmAttribute::kM;
    static constexpr index_t kN = WarpGemmAttribute::kN;
    static constexpr index_t kK = WarpGemmAttribute::kK;

    using ADataType = typename WarpGemmAttribute::ADataType;
    using BDataType = typename WarpGemmAttribute::BDataType;
    using CDataType = typename WarpGemmAttribute::CDataType;

    using AWarpDstrEncoding = typename WarpGemmAttribute::AWarpDstrEncoding;
    using BWarpDstrEncoding = typename WarpGemmAttribute::BWarpDstrEncoding;
    using CWarpDstrEncoding = typename WarpGemmAttribute::CWarpDstrEncoding;

    using AWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(AWarpDstrEncoding{}))>;
    using BWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(BWarpDstrEncoding{}))>;
    using CWarpDstr = remove_cvref_t<decltype(make_static_tile_distribution(CWarpDstrEncoding{}))>;

    using AWarpTensor = StaticDistributedTensor<ADataType, AWarpDstr>;
    using BWarpTensor = StaticDistributedTensor<BDataType, BWarpDstr>;
    using CWarpTensor = StaticDistributedTensor<CDataType, CWarpDstr>;

    __device__ void operator()(CWarpTensor& c, const AWarpTensor& a, const BWarpTensor& b) const
    {
        using AVec = typename vector_type<ADataType, AWarpTensor::GetThreadBufferSize()>::type;
        using BVec = typename vector_type<BDataType, BWarpTensor::GetThreadBufferSize()>::type;
        using CVec = typename vector_type<CDataType, CWarpTensor::GetThreadBufferSize()>::type;

        constexpr auto I0 = Number<0>{};

        const auto a_vec = a.GetThreadBuffer().template GetAsType<AVec>(I0);
        const auto b_vec = b.GetThreadBuffer().template GetAsType<BVec>(I0);
        auto c_vec       = c.GetThreadBuffer().template GetAsType<CVec>(I0);

        // c_vec += a_vec * b_vec
        WarpGemmAttribute{}(c_vec, a_vec, b_vec);

        c.GetThreadBuffer().template SetAsType<CVec>(I0, c_vec);
    }

    __device__ auto operator()(const AWarpTensor& a, const BWarpTensor& b) const
    {
        CWarpTensor c;

        using AVec = typename vector_type<ADataType, AWarpTensor::GetThreadBufferSize()>::type;
        using BVec = typename vector_type<BDataType, BWarpTensor::GetThreadBufferSize()>::type;
        using CVec = typename vector_type<CDataType, CWarpTensor::GetThreadBufferSize()>::type;

        constexpr auto I0 = Number<0>{};

        const auto a_vec = a.GetThreadBuffer().template GetAsType<AVec>(I0);
        const auto b_vec = b.GetThreadBuffer().template GetAsType<BVec>(I0);

        // c_vec = a_vec * b_vec
        auto c_vec = WarpGemmAttribute{}(a_vec, b_vec);

        c.GetThreadBuffer().template SetAsType<CVec>(I0, c_vec);

        return c;
    }
};

} // namespace warp
} // namespace tile_program
} // namespace ck
