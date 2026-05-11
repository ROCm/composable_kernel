#include <iostream>
#include <map>
#include <set>
#include "ck_tile/core.hpp"

using namespace ck_tile;

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
    
    std::vector<int> phase0 = {0, 1, 2, 3, 20, 21, 22, 23};
    
    int total_conflicts = 0;
    
    // Test all 32 columns, dm=0, Phase 0 only
    for(int k = 0; k < 32; k++) {
        std::map<int, std::set<int>> bank_to_slots;
        std::map<int, int> bank_to_count;
        
        for(int lane : phase0) {
            int m0_idx = lane / 8;
            int m = m0_idx * 8;  // dm=0
            
            auto offset = lds_desc.calculate_offset(make_tuple(k, m));
            int byte_offset = offset * DataTypeSize;
            int slot = byte_offset / 4;
            int bank = slot % 32;
            
            bank_to_slots[bank].insert(slot);
            bank_to_count[bank]++;
        }
        
        int conflicts_this_k = 0;
        for(const auto& [bank, count] : bank_to_count) {
            if(count > 1) {
                const auto& slots = bank_to_slots[bank];
                if(slots.size() > 1) {
                    // Different slots → conflict
                    conflicts_this_k += (count - 1);
                }
            }
        }
        
        if(k < 8) {
            std::cout << "k=" << k << ": ";
            for(const auto& [bank, count] : bank_to_count) {
                std::cout << "bank " << bank << " (" << count << " threads, " 
                          << bank_to_slots[bank].size() << " slots) ";
            }
            std::cout << "→ " << conflicts_this_k << " conflicts\n";
        }
        
        total_conflicts += conflicts_this_k;
    }
    
    std::cout << "\nTotal conflicts for Phase 0, dm=0, all 32 columns: " << total_conflicts << "\n";
    std::cout << "Average per column: " << (double)total_conflicts / 32 << "\n";
    std::cout << "\nExpected for full tile (32 cols × 8 dm × 8 phases): " 
              << total_conflicts * 8 * 8 << "\n";
    std::cout << "Profiler shows: 768\n";
    
    return 0;
}
