
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

namespace ck {

  template <ck::index_t M2, ck::index_t M4, ck::index_t CShuffleMXdlPerWavePerShuffle, ck::index_t CShuffleNXdlPerWavePerShuffle>
  struct PackedCast
  {
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

      // Calculate total elements in this slice
      constexpr index_t elements_per_slice = 
          CShuffleMXdlPerWavePerShuffle * CShuffleNXdlPerWavePerShuffle * M2 * M4;
      
      constexpr auto calculate_coords = [src_slice_origin_idx](auto idx) constexpr {

        // We know that the access order is 
        // Sequence<0, 1, 2, 3, 4, 5, 6, 7>,
        // Sequence<CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle, 1, 1, M2, 1, M4, 1>

        constexpr index_t m4_offset = idx.value % M4;
        constexpr index_t m2_offset = (idx.value / M4) % M2;
        constexpr index_t n_xdl_offset = (idx.value / (M4 * M2)) % CShuffleNXdlPerWavePerShuffle;
        constexpr index_t m_xdl_offset = idx.value / (M4 * M2 * CShuffleNXdlPerWavePerShuffle);
        
        return make_tuple(
            src_slice_origin_idx[Number<0>{}] + Number<m_xdl_offset>{},
            src_slice_origin_idx[Number<1>{}] + Number<n_xdl_offset>{},
            Number<0>{}, // this dim has unit size
            Number<0>{}, // this dim has unit size
            src_slice_origin_idx[Number<4>{}] + Number<m2_offset>{},
            Number<0>{}, // this dim has unit size
            src_slice_origin_idx[Number<6>{}] + Number<m4_offset>{},
            Number<0>{}  // this dim has unit size
        );
      };

      constexpr index_t num_pairs = elements_per_slice / 2;
      constexpr bool has_odd_element = (elements_per_slice % 2 == 1);

      static_for<0, num_pairs, 1>{}([&](auto pair_idx) {
          constexpr auto idx_0 = Number<pair_idx * 2>{};
          constexpr auto idx_1 = Number<pair_idx * 2 + 1>{};
          
          constexpr auto coord_0 = calculate_coords(idx_0);
          constexpr auto coord_1 = calculate_coords(idx_1);

          constexpr auto offset_0 = src_desc.CalculateOffset(coord_0);
          constexpr auto offset_1 = src_desc.CalculateOffset(coord_1);
          
          float& val_0 = src_buf(Number<offset_0>{});
          float& val_1 = src_buf(Number<offset_1>{});
          
          static_cast_float_to_bhalf_packed(val_0, val_1);
      });

      // Handle last element if the number of elements is odd.
      if constexpr (has_odd_element)
      {
          constexpr auto last_idx = Number<elements_per_slice - 1>{};
          constexpr auto last_coord = calculate_coords(last_idx);
          
          // Single element conversion
          constexpr auto last_offset = src_desc.CalculateOffset(last_coord);
          float& last_val = src_buf[Number<last_offset>{}];
          const auto single_bf16 = static_cast<__bf16>(last_val);
          uint16_t* parts = reinterpret_cast<uint16_t*>(&last_val);
          const uint16_t* bf16_bits = reinterpret_cast<const uint16_t*>(&single_bf16);
          parts[1] = bf16_bits[0];
      }

    };
  };


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
                                                  DstScalarPerAccess>;

      static_assert(SpaceFillingCurve::ScalarPerVector == 1,
                      "wrong! SpaceFillingCurve::ScalarPerVector must be 1 for PackedCastV2");

      constexpr index_t num_access = SpaceFillingCurve::GetNumOfAccess();
      constexpr index_t num_pairs = num_access / 2;
      constexpr bool has_odd_element = (num_access % 2 == 1);

      static_for<0, num_pairs, 1>{}([&](auto i_pair) 
      {
        constexpr auto idx_1d_0 = I2 * i_pair;
        constexpr auto idx_1d_1 = I2 * i_pair + I1;
        constexpr auto idx_md_0 = SpaceFillingCurve::GetIndex(idx_1d_0);
        constexpr auto idx_md_1 = SpaceFillingCurve::GetIndex(idx_1d_1);
    
        constexpr index_t src_offset_0 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_0);
        constexpr index_t src_offset_1 = src_desc.CalculateOffset(src_slice_origin_idx + idx_md_1);

        float& val_0 = src_buf(Number<src_offset_0>{});
        float& val_1 = src_buf(Number<src_offset_1>{});     
        static_cast_float_to_bhalf_packed_v2(val_0, val_1);
      });

      // Handle last element if the number of elements is odd.
      if constexpr (has_odd_element)
      {
          constexpr auto last_idx_1d = Number<num_access - 1>{};
          constexpr auto last_idx_md = SpaceFillingCurve::GetIndex(last_idx_1d);
          
          // Single element conversion
          constexpr auto last_src_offset = src_desc.CalculateOffset(last_idx_1d);
          float& last_val = src_buf(Number<last_src_offset>{});
          const auto single_bf16 = static_cast<__bf16>(last_val);
          uint16_t* parts = reinterpret_cast<uint16_t*>(&last_val);
          const uint16_t* bf16_bits = reinterpret_cast<const uint16_t*>(&single_bf16);
          parts[0] = bf16_bits[0];
      }
    };
  };
}
