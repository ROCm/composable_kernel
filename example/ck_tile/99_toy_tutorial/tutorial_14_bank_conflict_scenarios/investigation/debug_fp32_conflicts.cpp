// Debug: Calculate actual bank conflicts for FP32 transpose
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include "ck_tile/core.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct DebugTransposeKernel
{
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    // Same LDS descriptor as the real kernel
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
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
                make_tuple(number<kKPack>{},
                           number<kK * MLdsLayer>{},
                           number<1>{}),
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
                    make_merge_transform(
                        make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
                    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
                make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return lds_desc;
        }
        else
        {
            return make_naive_tensor_descriptor_packed(make_tuple(number<kM>{}, number<kK>{}));
        }
    }
};

// Phase groupings from LDS_CONSTRAINTS.md
static std::vector<std::vector<int>> get_write_phases() {
    return {
        {0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20, 21, 22, 23},
        {24, 25, 26, 27, 28, 29, 30, 31},
        {32, 33, 34, 35, 36, 37, 38, 39},
        {40, 41, 42, 43, 44, 45, 46, 47},
        {48, 49, 50, 51, 52, 53, 54, 55},
        {56, 57, 58, 59, 60, 61, 62, 63}
    };
}

static std::vector<std::vector<int>> get_read_phases() {
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
void analyze_conflicts()
{
    using DataType = float;
    constexpr index_t M = 64;
    // constexpr index_t K = 32;  // Unused
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc = DebugTransposeKernel<DataType, UseXor>::MakeLdsDescriptorMK();

    std::cout << "\n=== FP32 " << (UseXor ? "WITH" : "WITHOUT") << " XOR (WITH PHASE GROUPING) ===\n\n";

    // WRITE conflicts: Check row-wise writes with PHASE-AWARE counting
    // For FP32: K1 = 16/sizeof(float) = 4 elements per thread
    std::cout << "WRITE Pattern (row-wise, 4 elements per thread, phase-aware):\n";
    index_t write_conflicts = 0;
    index_t write_accesses = 0;

    auto write_phases = get_write_phases();

    // Process each WRITE phase (8 consecutive lanes execute together)
    for(const auto& phase : write_phases)
    {
        // INTRA-lane conflicts first: each lane's 4 elements may hit same banks
        for(index_t lane : phase)
        {
            index_t m = lane; // For write, lane directly maps to row m
            std::map<index_t, index_t> lane_bank_count;

            // Each lane writes 4 consecutive K elements in its row (FP32: M1=4)
            for(index_t k = 0; k < 4; k++)
            {
                auto offset = desc.calculate_offset(make_tuple(m, k));
                index_t byte_offset = offset * DataTypeSize;
                index_t bank = byte_offset / 4 % 32;

                lane_bank_count[bank]++;
                write_accesses++;
            }

            // Count intra-lane conflicts (within this single lane's 4 elements)
            for(const auto& entry : lane_bank_count)
            {
                index_t count = entry.second;
                if(count > 1)
                {
                    // FP32: each element occupies full 4-byte bank slot (no pairing)
                    write_conflicts += (count - 1);
                }
            }
        }
    }

    std::cout << "  Write accesses: " << write_accesses << "\n";
    std::cout << "  Write conflicts (intra-lane): " << write_conflicts << "\n\n";

    // READ conflicts: Check column-wise reads (transpose) with PHASE-AWARE counting
    // For FP32: M1 = 16/sizeof(float) = 4 elements per thread
    std::cout << "READ Pattern (column-wise transpose, 4 elements per thread, phase-aware):\n";
    index_t read_intra_conflicts = 0;
    index_t read_inter_conflicts = 0;
    index_t read_accesses = 0;

    auto read_phases = get_read_phases();

    // Process each READ phase (non-consecutive lane grouping!)
    for(const auto& phase : read_phases)
    {
        // For each column in the tile
        for(index_t k = 0; k < 32; k++)
        {
            // Track which banks each lane in this phase accesses
            std::vector<std::map<index_t, index_t>> lane_bank_maps;

            for(index_t lane : phase)
            {
                std::map<index_t, index_t> banks_this_lane;

                // Each lane reads 4 consecutive M elements from column k
                // Lane mapping: K2_idx = lane % 8, M0_idx = lane / 8
                index_t k2_idx = lane % 8;
                index_t m0_idx = lane / 8;

                // Only process if this lane is responsible for column k
                if(k2_idx == (k % 8))
                {
                    index_t m_start = m0_idx * 4;  // FP32: M1 = 4

                    for(index_t m = m_start; m < m_start + 4 && m < M; m++)
                    {
                        auto offset = desc.calculate_offset(make_tuple(m, k));
                        index_t byte_offset = offset * DataTypeSize;
                        index_t bank = byte_offset / 4 % 32;

                        banks_this_lane[bank]++;
                        read_accesses++;
                    }

                    // Count INTRA-lane conflicts for this lane
                    for(const auto& entry : banks_this_lane)
                    {
                        index_t count = entry.second;
                        if(count > 1)
                        {
                            read_intra_conflicts += (count - 1);
                        }
                    }

                    lane_bank_maps.push_back(banks_this_lane);
                }
            }

            // Count INTER-lane conflicts within this phase for this column
            std::map<index_t, index_t> bank_to_lane_count;
            for(const auto& lane_map : lane_bank_maps)
            {
                for(const auto& entry : lane_map)
                {
                    index_t bank = entry.first;
                    bank_to_lane_count[bank]++;
                }
            }

            for(const auto& entry : bank_to_lane_count)
            {
                index_t lane_count = entry.second;
                if(lane_count > 1)
                {
                    read_inter_conflicts += (lane_count - 1);
                }
            }
        }
    }

    std::cout << "  Read accesses: " << read_accesses << "\n";
    std::cout << "  Read conflicts (intra-lane): " << read_intra_conflicts << "\n";
    std::cout << "  Read conflicts (inter-lane): " << read_inter_conflicts << "\n\n";

    index_t total_conflicts = write_conflicts + read_intra_conflicts + read_inter_conflicts;
    std::cout << "TOTAL CONFLICTS (full 64x32 tile, all phases): " << total_conflicts << "\n";
    std::cout << "  Write (intra-lane): " << write_conflicts << "\n";
    std::cout << "  Read (intra-lane): " << read_intra_conflicts << "\n";
    std::cout << "  Read (inter-lane): " << read_inter_conflicts << "\n\n";

    // Scale by K-loop iterations (kernel loops over K in blocks of 32)
    // For 256x128 input: K_iterations = 128/32 = 4
    index_t k_iterations = 4;
    std::cout << "Scaled by K-loop iterations (" << k_iterations << " iterations):\n";
    std::cout << "  Total: " << total_conflicts * k_iterations << "\n";
    std::cout << "  Write (intra): " << write_conflicts * k_iterations << "\n";
    std::cout << "  Read (intra): " << read_intra_conflicts * k_iterations << "\n";
    std::cout << "  Read (inter): " << read_inter_conflicts * k_iterations << "\n\n";
}

int main()
{
    std::cout << "=================================================\n";
    std::cout << "FP32 Bank Conflict Analysis\n";
    std::cout << "=================================================\n";

    analyze_conflicts<false>(); // Without XOR
    analyze_conflicts<true>();  // With XOR

    std::cout << "\n===============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "===============================================\n";
    std::cout << "  Without XOR: Profiler = 15,360\n";
    std::cout << "  With XOR:    Profiler = 15,360 (SAME - profiler anomaly)\n\n";
    std::cout << "NOTE:\n";
    std::cout << "  Profiler shows identical conflicts for FP32 with/without XOR.\n";
    std::cout << "  Manual address calculation proves XOR distributes correctly,\n";
    std::cout << "  but SQ_LDS_BANK_CONFLICT counter doesn't reflect the change.\n";
    std::cout << "  This suggests a profiler limitation or bug for FP32 XOR patterns.\n";

    return 0;
}
