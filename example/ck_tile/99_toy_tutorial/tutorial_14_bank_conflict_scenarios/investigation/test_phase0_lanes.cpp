#include <iostream>
#include "ck_tile/core.hpp"

using namespace ck_tile;

int main() {
    constexpr index_t kM = 64;
    constexpr index_t kK = 32;
   
    auto desc_km = make_naive_tensor_descriptor(
        make_tuple(number<kK>{}, number<kM>{}),
        make_tuple(number<1>{}, number<kK>{}));
    
    std::vector<int> phase0 = {0, 1, 2, 3, 20, 21, 22, 23};
    
    std::cout << "Phase 0 lanes accessing column k=0, dm=0:\n";
    for(int lane : phase0) {
        int m0_idx = lane / 8;
        int m = m0_idx * 8;  // dm=0
        
        auto offset = desc_km.calculate_offset(make_tuple(0, m));
        int byte_offset = offset * 2;
        int slot = byte_offset / 4;
        int bank = slot % 32;
        
        std::cout << "  Lane " << lane << " → m=" << m << " → slot=" << slot << " → bank=" << bank << "\n";
    }
    
    return 0;
}
