// Simple: Print which bank each lane accesses, per phase, using actual XOR descriptor

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

constexpr index_t kM = 64;
constexpr index_t kK = 32;
constexpr index_t kKPack = 8;

// XOR descriptor from 04_row_major_xor.cpp
CK_TILE_HOST_DEVICE static constexpr auto MakeXorDescriptor()
{
    using DataType = half_t;
    constexpr auto DataTypeSize = sizeof(DataType);
    constexpr auto MLdsLayer = (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

    constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kK / kKPack * MLdsLayer>{},
                   number<kM / MLdsLayer>{},
                   number<kKPack>{}),
        make_tuple(number<kKPack>{}, number<kK * MLdsLayer>{}, number<1>{}),
        number<kKPack>{},
        number<1>{});

    constexpr auto lds_desc_permuted = transform_tensor_descriptor(
        lds_desc_0,
        make_tuple(make_xor_transform(make_tuple(number<kM / MLdsLayer>{},
                                                 number<kK / kKPack * MLdsLayer>{})),
                   make_pass_through_transform(number<kKPack>{})),
        make_tuple(sequence<1, 0>{}, sequence<2>{}),
        make_tuple(sequence<1, 0>{}, sequence<2>{}));

    constexpr auto lds_desc_unmerged = transform_tensor_descriptor(
        lds_desc_permuted,
        make_tuple(make_unmerge_transform(
                       make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
                   make_pass_through_transform(number<kM / MLdsLayer>{}),
                   make_pass_through_transform(number<kKPack>{})),
        make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
        make_tuple(sequence<0, 2>{}, sequence<1>{}, sequence<3>{}));

    constexpr auto lds_desc = transform_tensor_descriptor(
        lds_desc_unmerged,
        make_tuple(
            make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
            make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return lds_desc;
}

int main()
{
    constexpr auto xor_desc = MakeXorDescriptor();

    std::cout << "=== XOR Descriptor: Bank per (m, k) ===\n\n";

    // Thread distribution for KM read (transpose):
    // k = wf * 8 + lane / 8
    // m = (lane % 8) * 8 + dm

    int wf = 0;  // WF0
    int dm = 0;  // First read

    std::cout << "WF=" << wf << ", dm=" << dm << "\n\n";

    for (int phase = 0; phase < 8; phase++)
    {
        std::cout << "Phase " << phase << " (lanes " << (phase * 8) << "-" << (phase * 8 + 7) << "):\n";
        std::cout << "  Lane |  k |  m | offset | bank\n";
        std::cout << "  -----|----|----|--------|-----\n";

        std::map<int, std::vector<int>> bank_to_lanes;

        for (int lane_in_phase = 0; lane_in_phase < 8; lane_in_phase++)
        {
            int lane = phase * 8 + lane_in_phase;

            int k = wf * 8 + lane / 8;
            int m = (lane % 8) * 8 + dm;

            auto offset = xor_desc.calculate_offset(make_multi_index(m, k));
            int byte_addr = offset * sizeof(half_t);
            int bank = (byte_addr / 4) % 32;

            std::cout << "    " << lane << "  |  " << k << " | " << m
                      << " |   " << offset << " |  " << bank << "\n";

            bank_to_lanes[bank].push_back(lane);
        }

        std::cout << "  Summary: ";
        for (const auto& [bank, lanes] : bank_to_lanes)
        {
            std::cout << "B" << bank << "(" << lanes.size() << ") ";
        }
        int conflicts = 0;
        for (const auto& [bank, lanes] : bank_to_lanes)
        {
            if (lanes.size() > 1) conflicts += lanes.size() - 1;
        }
        std::cout << "-> " << conflicts << " conflicts\n\n";
    }

    // Also show all 8 dm values for phase 0
    std::cout << "=== Phase 0 across all dm values ===\n\n";
    for (int dm_val = 0; dm_val < 8; dm_val++)
    {
        std::cout << "dm=" << dm_val << ": ";
        std::map<int, int> bank_count;

        for (int lane = 0; lane < 8; lane++)
        {
            int k = wf * 8 + lane / 8;  // All lanes in phase 0 have k = 0
            int m = (lane % 8) * 8 + dm_val;

            auto offset = xor_desc.calculate_offset(make_multi_index(m, k));
            int byte_addr = offset * sizeof(half_t);
            int bank = (byte_addr / 4) % 32;

            bank_count[bank]++;
        }

        for (const auto& [bank, count] : bank_count)
        {
            std::cout << "B" << bank << "(" << count << ") ";
        }

        int conflicts = 0;
        for (const auto& [bank, count] : bank_count)
        {
            if (count > 1) conflicts += count - 1;
        }
        std::cout << "-> " << conflicts << " conflicts\n";
    }

    return 0;
}
