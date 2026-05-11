#include <iostream>
#include <map>
#include <set>
#include <vector>
#include "ck_tile/core.hpp"

using namespace ck_tile;

std::vector<std::vector<int>> get_phases() {
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

int main() {
    constexpr index_t kM = 64;
    constexpr index_t kK = 32;
    constexpr index_t kKPack = 8;
    constexpr index_t DataTypeSize = 2;
    
    constexpr auto MLdsLayer = (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

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
            make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{})),
            make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{}))),
        make_tuple(sequence<2, 3>{}, sequence<1, 0>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));
    
    auto phases = get_phases();
    
    int total_conflicts = 0;
    
    // Column k=0, dm=0, ALL phases
    std::cout << "XOR: Column k=0, dm=0, checking all phases:\n";
    for(size_t phase_idx = 0; phase_idx < phases.size(); phase_idx++) {
        const auto& phase = phases[phase_idx];
        std::map<int, std::set<int>> bank_to_slots;
        std::map<int, int> bank_to_count;
        
        for(int lane : phase) {
            int m0_idx = lane / 8;
            int m = m0_idx * 8;  // dm=0
            
            auto offset = lds_desc.calculate_offset(make_tuple(0, m));
            int byte_offset = offset * DataTypeSize;
            int slot = byte_offset / 4;
            int bank = slot % 32;
            
            bank_to_slots[bank].insert(slot);
            bank_to_count[bank]++;
        }
        
        int conflicts_this_phase = 0;
        for(const auto& [bank, count] : bank_to_count) {
            if(count > 1) {
                const auto& slots = bank_to_slots[bank];
                if(slots.size() > 1) {
                    conflicts_this_phase += (count - 1);
                }
            }
        }
        
        std::cout << "  Phase " << phase_idx << ": ";
        for(const auto& [bank, count] : bank_to_count) {
            std::cout << "bank" << bank << "(" << count << "t," << bank_to_slots[bank].size() << "s) ";
        }
        std::cout << "→ " << conflicts_this_phase << " conflicts\n";
        
        total_conflicts += conflicts_this_phase;
    }
    
    std::cout << "\nTotal for k=0, dm=0, all phases: " << total_conflicts << "\n";
    std::cout << "Expected for full tile (×32 cols ×8 dm): " << total_conflicts * 32 * 8 << "\n";
    std::cout << "Profiler: 768\n";
    
    return 0;
}
