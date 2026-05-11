// Test INTRA-LANE conflicts
// When a single thread does a vector load (ds_read_b128 for 8 FP16),
// it accesses 8 consecutive addresses. If any of these hit the same bank
// multiple times, that causes intra-lane conflicts!
#include <iostream>
#include <set>
#include <map>
#include <vector>
#include "ck_tile/core.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct TestDescriptors
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

// Check intra-lane conflicts for a single lane's vector load
// A lane loads 8 consecutive M values: m_start, m_start+1, ..., m_start+7
template<bool UseXor>
index_t count_intra_lane_conflicts(int lane, int k)
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    int m0_idx = lane / 8;
    int m_start = m0_idx * 8;

    std::map<index_t, int> bank_hits;

    for (int m1 = 0; m1 < 8; m1++) {
        int m = m_start + m1;
        auto offset = desc_km.calculate_offset(make_tuple(k, m));
        index_t byte_offset = offset * DataTypeSize;
        index_t slot = byte_offset / 4;
        index_t bank = slot % 32;
        bank_hits[bank]++;
    }

    // Count conflicts: for each bank hit N times, add (N-1) conflicts
    index_t conflicts = 0;
    for (const auto& [bank, count] : bank_hits) {
        if (count > 1) {
            conflicts += (count - 1);
        }
    }
    return conflicts;
}

template<bool UseXor>
void analyze_intra_lane()
{
    std::cout << "\n=== INTRA-LANE CONFLICTS (" << (UseXor ? "XOR" : "PLAIN") << ") ===\n\n";

    // Check lane 0 for all k values
    std::cout << "Lane 0 (M0_idx=0, accesses m=0-7):\n";
    for (int k = 0; k < 32; k++) {
        index_t conf = count_intra_lane_conflicts<UseXor>(0, k);
        if (conf > 0) {
            std::cout << "  k=" << k << ": " << conf << " conflicts\n";
        }
    }

    // Total across all lanes and all k iterations
    index_t total = 0;
    for (int lane = 0; lane < 64; lane++) {
        // K1 iterations (4)
        for (int k1 = 0; k1 < 4; k1++) {
            int k2_idx = lane % 8;
            int k = k1 * 8 + k2_idx;
            total += count_intra_lane_conflicts<UseXor>(lane, k);
        }
    }

    std::cout << "\nTotal intra-lane conflicts (all lanes, all K1): " << total << "\n";
    std::cout << "Scaled (4 blocks): " << total * 4 << "\n";
}

// Debug: show bank pattern for lane 0's vector load at k=0
template<bool UseXor>
void debug_lane0_vector_load()
{
    using DataType = half_t;
    constexpr index_t DataTypeSize = sizeof(DataType);

    constexpr auto desc_km = TestDescriptors<DataType, UseXor>::MakeLdsDescriptorKM();

    std::cout << "\n=== Lane 0 vector load at k=0 (" << (UseXor ? "XOR" : "PLAIN") << ") ===\n";
    std::cout << "Accessing m=0,1,2,3,4,5,6,7:\n";

    for (int m = 0; m < 8; m++) {
        auto offset = desc_km.calculate_offset(make_tuple(0, m));
        index_t byte_offset = offset * DataTypeSize;
        index_t slot = byte_offset / 4;
        index_t bank = slot % 32;
        std::cout << "  m=" << m << " → offset=" << offset << ", byte=" << byte_offset
                  << ", slot=" << slot << ", bank=" << bank << "\n";
    }
}

int main()
{
    std::cout << "=============================================\n";
    std::cout << "INTRA-LANE Conflict Analysis\n";
    std::cout << "=============================================\n";
    std::cout << "\nKey insight: A single thread's vector load accesses\n";
    std::cout << "8 consecutive M values. If these hit the same bank\n";
    std::cout << "multiple times, that's an intra-lane conflict!\n";

    debug_lane0_vector_load<false>();
    debug_lane0_vector_load<true>();

    analyze_intra_lane<false>();
    analyze_intra_lane<true>();

    std::cout << "\n=============================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "=============================================\n";

    return 0;
}
