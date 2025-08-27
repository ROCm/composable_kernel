
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/static_buffer.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {
  template <ck::index_t M2, ck::index_t M4, ck::index_t CShuffleMXdlPerWavePerShuffle, ck::index_t CShuffleNXdlPerWavePerShuffle>
  struct PackedCastV2
  {
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    template <typename SrcDesc, typename SrcSliceOriginIdx, typename SrcBuffer>
    __device__ void Run(const SrcDesc&, const SrcSliceOriginIdx&, SrcBuffer& src_buf)
    {
      static_assert(SrcDesc::IsKnownAtCompileTime(),
                        "wrong! SrcDesc need to known at compile-time");
      static_assert(is_known_at_compile_time<remove_cvref_t<SrcSliceOriginIdx>>::value,
                    "wrong! SrcSliceOrigin need to known at compile-time");

      static_assert(SrcBuffer::IsStaticBuffer(), "wrong! SrcBuffer need to be StaticBuffer");

      constexpr auto src_desc             = remove_cvref_t<SrcDesc>{};
      constexpr auto src_slice_origin_idx = to_multi_index(SrcSliceOriginIdx{});

      using SliceLengths = Sequence<CShuffleMXdlPerWavePerShuffle,
                                    CShuffleNXdlPerWavePerShuffle,
                                    I1,
                                    I1,
                                    M2,
                                    I1,
                                    M4,
                                    I1>;
      using DimAccessOrder = Sequence<0, 1, 2, 3, 4, 5, 6, 7>;
      using DstScalarPerAccess = Sequence<1, 1, 1, 1, 1, 1, 1, 1>;
      using SpaceFillingCurve = SpaceFillingCurve<SliceLengths,
                                                  DimAccessOrder,
                                                  DstScalarPerAccess,
                                                  false>;

      static_assert(SpaceFillingCurve::ScalarPerVector == 1,
                      "wrong! SpaceFillingCurve::ScalarPerVector must be 1 for PackedCastV2");

      constexpr index_t num_access = SpaceFillingCurve::GetNumOfAccess();
      constexpr index_t num_pairs = num_access / 2;
      constexpr bool has_odd_element = (num_access % 2 == 1);

      static_assert(!has_odd_element, "PackedCastV2 does not support odd number of elements");

      ck::float2_t float2_buffer;
      static_for<0, num_pairs, 1>{}([&](auto i_pair) 
      {
        constexpr auto idx_1d_0 = I2 * i_pair;
        constexpr auto idx_1d_1 = I2 * i_pair + I1;
        constexpr auto idx_md_0 = SpaceFillingCurve::GetIndex(idx_1d_0);
        constexpr auto idx_md_1 = SpaceFillingCurve::GetIndex(idx_1d_1);
        constexpr auto idx_md_pair = SpaceFillingCurve::GetIndex(i_pair);
    
        constexpr index_t src_offset_0 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_0);
        constexpr index_t src_offset_1 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_1);
        constexpr index_t pair_offset = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_pair);

        if constexpr (src_offset_1 - src_offset_0 == 1)
        {
            // Load two consecutive float values from the src buffer
            float2_buffer = src_buf.template GetAsType<ck::float2_t>(Number<src_offset_0>{});
        }
        else 
        {
            // Load the two float values one by one
            float2_buffer= {src_buf[Number<src_offset_0>{}], src_buf[Number<src_offset_1>{}]};
        }

        // Store the packed bfloat2 value back to the src buffer
        const ck::bhalf2_t packed_value= bf16x2_convert_rne<ck::bhalf2_t, float>(float2_buffer[0], float2_buffer[1]);
        union {
            ck::bhalf2_t bhalf2;
            float fp32;
        } converter;
        converter.bhalf2 = packed_value;
        src_buf(Number<pair_offset>{}) = converter.fp32;
      });
    };
  };
}
