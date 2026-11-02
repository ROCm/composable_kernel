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

} // namespace ck_tile
