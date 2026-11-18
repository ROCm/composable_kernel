// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_softmax.hpp"

namespace ck_tile {

template <index_t ColumnGroups>
struct FmhaTdmSchedOutputFragmentsFor
{
    using Mapping  = FmhaTdmSchedScoreFragmentMappingFor<ColumnGroups>;
    using Fragment = fp32x8_t;

    static constexpr index_t kNumDmsb             = 4;
    static constexpr index_t kNumN                = ColumnGroups;
    static constexpr index_t kNumFragments        = kNumDmsb * kNumN;
    static constexpr index_t kElementsPerFragment = 8;

    template <typename Initializer>
    CK_TILE_HOST_DEVICE static constexpr auto Make(Initializer&& initializer)
    {
        return generate_tuple([&](auto ordinal) { return initializer(ordinal); },
                              number<kNumFragments>{});
    }

    CK_TILE_HOST_DEVICE static constexpr auto MakeZero()
    {
        return Make([](auto) { return Fragment{}; });
    }

    template <index_t Ordinal, index_t Element>
    CK_TILE_HOST_DEVICE static constexpr index_t GetThreadBufferOffset()
    {
        static_assert(Ordinal >= 0 && Ordinal < kNumFragments);
        static_assert(Element >= 0 && Element < kElementsPerFragment);
        constexpr index_t d_msb = Ordinal / kNumN;
        constexpr index_t n     = Ordinal % kNumN;
        constexpr index_t pair  = n * Mapping::kPairsPerFragment + Element / 2;
        return Mapping::GetThreadBufferOffset(d_msb, pair, Element % 2);
    }

    CK_TILE_HOST_DEVICE static constexpr bool ValidateMapping()
    {
        bool seen[Mapping::kThreadBufferSize] = {};
        for(index_t ordinal = 0; ordinal < kNumFragments; ++ordinal)
        {
            const index_t d_msb = ordinal / kNumN;
            const index_t n     = ordinal % kNumN;
            for(index_t element = 0; element < kElementsPerFragment; ++element)
            {
                const index_t pair   = n * Mapping::kPairsPerFragment + element / 2;
                const index_t offset = Mapping::GetThreadBufferOffset(d_msb, pair, element % 2);
                if(offset < 0 || offset >= Mapping::kThreadBufferSize || seen[offset])
                {
                    return false;
                }
                seen[offset] = true;
            }
        }
        for(bool value : seen)
        {
            if(!value)
            {
                return false;
            }
        }
        return true;
    }

    template <typename OutputTensor, typename Fragments>
    CK_TILE_HOST_DEVICE static constexpr auto Reconstruct(const Fragments& fragments)
    {
        static_assert(OutputTensor::get_thread_buffer_size() == Mapping::kThreadBufferSize);
        auto output = OutputTensor{};
        static_ford<sequence<kNumFragments, kElementsPerFragment>>{}([&](auto indices) {
            constexpr index_t ordinal = indices[number<0>{}];
            constexpr index_t element = indices[number<1>{}];
            constexpr index_t offset  = GetThreadBufferOffset<ordinal, element>();
            output.get_thread_buffer()[number<offset>{}] = fragments.at(number<ordinal>{})[element];
        });
        return output;
    }
};

using FmhaTdmSchedOutputFragments = FmhaTdmSchedOutputFragmentsFor<4>;

static_assert(FmhaTdmSchedOutputFragments::kNumFragments == 16);
static_assert(FmhaTdmSchedOutputFragments::kNumFragments *
                  FmhaTdmSchedOutputFragments::kElementsPerFragment ==
              FmhaTdmSchedScoreFragmentMapping::kThreadBufferSize);
static_assert(FmhaTdmSchedOutputFragments::ValidateMapping());

// Each wave copies only the rows it staged, so no barrier follows the staging. The async store
// has no bounds check: rows past num_valid_rows are clamped to the wave's last valid row.
template <typename FmhaPipeline, index_t AsyncStoreBytes, typename ODataType, typename OAccTile>
CK_TILE_DEVICE void fmha_tdm_sched_store_o_tile_async_from_lds(void* o_lds,
                                                               ODataType* o_tile_ptr,
                                                               index_t stride_o,
                                                               index_t num_valid_rows,
                                                               const OAccTile& o_acc_tile)
{
    constexpr index_t kM              = FmhaPipeline::kM0;
    constexpr index_t kN              = FmhaPipeline::kN1;
    constexpr index_t kVec            = AsyncStoreBytes / sizeof(ODataType);
    constexpr index_t kLdsRowElems    = kN + kVec;
    constexpr index_t kStoresPerBurst = 8;
    static_assert(kM * kLdsRowElems * sizeof(ODataType) <= FmhaPipeline::Policy::kKFootprintBytes,
                  "O staging must fit in the K0 slot");
    static_assert(kLdsRowElems * sizeof(ODataType) % AsyncStoreBytes == 0);
    static_assert(FmhaPipeline::Policy::GetLdsOffsetK0() % AsyncStoreBytes == 0);

    // Wave w holds rows m * kMWarp * kWarpM + w * kWarpM + [0, kWarpM), m < kMIter.
    using OEncoding                 = typename OAccTile::StaticTileDistribution::DstrEncode;
    using OBlockWarps               = typename FmhaPipeline::BlockFmhaShape::Gemm1BlockWarps;
    constexpr index_t kMIter        = OEncoding::hs_lengthss_[number<0>{}][number<0>{}];
    constexpr index_t kMWarp        = OEncoding::hs_lengthss_[number<0>{}][number<1>{}];
    constexpr index_t kWarpM        = kM / (kMIter * kMWarp);
    constexpr index_t kLanes        = FmhaPipeline::Geometry::kWaveSize;
    constexpr index_t kChunks       = kN / kVec;
    constexpr index_t kRowsPerStore = kLanes / kChunks;
    static_assert(kMWarp == OBlockWarps::at(number<0>{}) && OBlockWarps::at(number<1>{}) == 1 &&
                      OEncoding::ps_to_rhss_major_[number<0>{}][number<0>{}] == 1 &&
                      OEncoding::ps_to_rhss_minor_[number<0>{}][number<0>{}] == 1,
                  "the wave index must select the M warp of the O tile");
    static_assert(kMIter * kMWarp * kWarpM == kM && kMWarp * kLanes == FmhaPipeline::kBlockSize);
    static_assert(kLanes % kChunks == 0 && kWarpM % kRowsPerStore == 0);

    // K0 is reused: wait for every wave's TDM writes and LDS reads of it.
    s_wait_tensorcnt<0>();
    block_sync_lds<0>();

    auto o_lds_view = make_tensor_view<address_space_enum::lds>(
        static_cast<ODataType*>(o_lds),
        make_naive_tensor_descriptor(make_tuple(number<kM>{}, number<kN>{}),
                                     make_tuple(number<kLdsRowElems>{}, number<1>{}),
                                     number<kVec>{},
                                     number<1>{}));
    auto o_lds_window = make_tile_window(o_lds_view,
                                         make_tuple(number<kM>{}, number<kN>{}),
                                         {0, 0},
                                         o_acc_tile.get_tile_distribution());
    store_tile(o_lds_window, cast_tile<ODataType>(o_acc_tile));
    s_wait_dscnt<0>();

    const index_t i_wave = get_warp_id();
    const index_t i_lane = get_lane_id();
    index_t last_row     = -1;
    static_for<0, kMIter, 1>{}([&](auto i_m) {
        const index_t first_row = decltype(i_m)::value * kMWarp * kWarpM + i_wave * kWarpM;
        if(num_valid_rows > first_row)
            last_row = min(num_valid_rows - 1, first_row + kWarpM - 1);
    });
    if(last_row >= 0)
    {
        constexpr index_t kStores = kMIter * kWarpM / kRowsPerStore;
        static_assert(kStores % kStoresPerBurst == 0);
        const index_t lane_row  = i_wave * kWarpM + i_lane / kChunks;
        const index_t col       = i_lane % kChunks * kVec;
        const uint32_t lds_base = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(o_lds));
        static_for<0, kStores / kStoresPerBurst, 1>{}([&](auto i_group) {
            uint32_t global_offset[kStoresPerBurst];
            uint32_t lds_addr[kStoresPerBurst];
            static_for<0, kStoresPerBurst, 1>{}([&](auto i_store) {
                constexpr index_t wave_row =
                    (decltype(i_group)::value * kStoresPerBurst + decltype(i_store)::value) *
                    kRowsPerStore;
                constexpr index_t tile_row =
                    wave_row / kWarpM * kMWarp * kWarpM + wave_row % kWarpM;
                const index_t row      = min(tile_row + lane_row, last_row);
                global_offset[i_store] = static_cast<uint32_t>(row) *
                                             static_cast<uint32_t>(stride_o * sizeof(ODataType)) +
                                         static_cast<uint32_t>(col * sizeof(ODataType));
                lds_addr[i_store] = lds_base + static_cast<uint32_t>((row * kLdsRowElems + col) *
                                                                     sizeof(ODataType));
            });
            amd_async_global_store_from_lds_b128_x8(o_tile_ptr, global_offset, lds_addr);
        });
    }
    s_wait_asynccnt<0>();
}

} // namespace ck_tile
