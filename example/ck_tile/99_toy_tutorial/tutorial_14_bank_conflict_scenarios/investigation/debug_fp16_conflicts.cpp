// Bank Conflict Calculator - WITH PHASES AND INTER-WAVEFRONT
//
// Key corrections:
// 1. Use actual phase groupings (which lanes execute together)
// 2. Consider all 4 wavefronts executing simultaneously
// 3. Count conflicts across all simultaneous accesses

#include <iostream>
#include <set>
#include <map>
#include <vector>
#include "ck_tile/core.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct LdsDescriptors
{
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        if constexpr (UseXor)
        {
            constexpr auto DataTypeSize = sizeof(DataType);
            constexpr auto MLdsLayer =
                (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

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
                    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{})),
                    make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{}))),
                make_tuple(sequence<2, 3>{}, sequence<1, 0>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return lds_desc;
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(number<1>{}, number<kK>{}));
        }
    }
};

// READ phase groupings - which lanes execute together within a wavefront
std::vector<std::vector<int>> get_read_phases()
{
    return {
        {0, 1, 2, 3, 20, 21, 22, 23},
        {4, 5, 6, 7, 16, 17, 18, 19},
        {8, 9, 10, 11, 28, 29, 30, 31},
        {12, 13, 14, 15, 24, 25, 26, 27},
        {32, 33, 34, 35, 52, 53, 54, 55},
        {36, 37, 38, 39, 48, 49, 50, 51},
        {40, 41, 42, 43, 60, 61, 62, 63},
        {44, 45, 46, 47, 56, 57, 58, 59}
    };
}

template<bool UseXor>
void analyze_bank_conflicts()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = LdsDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  " << (UseXor ? "WITH XOR   " : "WITHOUT XOR") << " - Phase + Inter-WF Analysis                  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    auto phases = get_read_phases();

    std::cout << "Execution model:\n";
    std::cout << "  - 4 wavefronts execute simultaneously\n";
    std::cout << "  - Within each WF, 8 phases execute sequentially\n";
    std::cout << "  - Each phase has 8 lanes executing together\n";
    std::cout << "  - Total per phase: 8 lanes × 4 WFs = 32 simultaneous accesses\n\n";

    index_t total_conflicts = 0;

    // For each phase (8 phases, sequential within WF)
    for (int phase_idx = 0; phase_idx < 8; phase_idx++)
    {
        const auto& phase_lanes = phases[phase_idx];
        index_t phase_conflicts = 0;

        std::cout << "Phase " << phase_idx << " lanes: {";
        for (size_t i = 0; i < phase_lanes.size(); i++)
        {
            if (i > 0) std::cout << ",";
            std::cout << phase_lanes[i];
        }
        std::cout << "}\n";

        // For each M1 step (8 scalar reads per thread)
        for (int m1 = 0; m1 < 8; m1++)
        {
            // All 4 wavefronts execute this phase's M1 step simultaneously
            // Collect all 32 accesses (8 lanes × 4 WFs)
            std::map<int, std::set<int>> bank_to_slots;
            std::map<int, int> bank_access_count;

            for (int wf = 0; wf < 4; wf++)
            {
                for (int lane : phase_lanes)
                {
                    int k2 = lane % 8;
                    int m0 = lane / 8;
                    int k = wf * 8 + k2;
                    int m = m0 * 8 + m1;

                    auto offset = desc_km.calculate_offset(make_tuple(k, m));
                    int byte_offset = offset * DataTypeSize;
                    int slot = byte_offset / 4;
                    int bank = slot % 32;

                    bank_access_count[bank]++;
                    bank_to_slots[bank].insert(slot);
                }
            }

            // Count conflicts for this (phase, m1) step
            for (auto& entry : bank_access_count)
            {
                int bank = entry.first;
                int unique_slots = bank_to_slots[bank].size();
                (void)bank; // Used in debug output

                // Conflicts = unique_slots - 1 (FP16 same-slot = 0 conflicts)
                if (unique_slots > 1)
                {
                    phase_conflicts += (unique_slots - 1);
                }
            }

            // Debug: show first M1 step for Phase 0
            if (phase_idx == 0 && m1 == 0)
            {
                std::cout << "\n  Debug: Phase 0, M1=0 (32 accesses from 4 WFs):\n";
                for (auto& entry : bank_access_count)
                {
                    int bank = entry.first;
                    int num_accesses = entry.second;
                    int unique_slots = bank_to_slots[bank].size();
                    int conflicts = unique_slots > 1 ? unique_slots - 1 : 0;

                    std::cout << "    Bank " << bank << ": " << num_accesses
                              << " accesses, " << unique_slots << " slots → "
                              << conflicts << " conflicts\n";
                }
                std::cout << "\n";
            }
        }

        std::cout << "  Phase " << phase_idx << " conflicts: " << phase_conflicts << "\n";
        total_conflicts += phase_conflicts;
    }

    std::cout << "\n─────────────────────────────────────\n";
    std::cout << "Total per tile (64×32): " << total_conflicts << "\n";

    // Scale by number of tiles (4 blocks × 1 K-iteration for 256×32 test)
    // Actually for 256×128: 4 M-blocks × 4 K-iterations = but profiler counts per kernel
    constexpr int num_blocks = 4;
    index_t scaled_conflicts = total_conflicts * num_blocks;

    std::cout << "Scaled (×" << num_blocks << " blocks): " << scaled_conflicts << "\n";
}

int main()
{
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  FP16 Bank Conflict Calculator - Phase + Inter-WF Model       ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    analyze_bank_conflicts<false>();
    analyze_bank_conflicts<true>();

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  PROFILER TARGETS                                             ║\n";
    std::cout << "║  WITHOUT XOR: 7,168                                           ║\n";
    std::cout << "║  WITH XOR:    3,072                                           ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";

    return 0;
}
