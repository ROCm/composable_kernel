#include <iostream>
#include "ck_tile/core.hpp"

using namespace ck_tile;

int main() {
    constexpr index_t kM = 64;
    constexpr index_t kK = 32;
   
    // Plain row-major descriptor for transpose READ [K, M]
    auto desc_km = make_naive_tensor_descriptor(
        make_tuple(number<kK>{}, number<kM>{}),
        make_tuple(number<1>{}, number<kK>{}));
    
    std::cout << "WITHOUT XOR - Column k=0 bank pattern:\n";
    for(int m = 0; m < 8; m++) {
        auto offset = desc_km.calculate_offset(make_tuple(0, m));
        int byte_offset = offset * 2;  // FP16
        int slot = byte_offset / 4;
        int bank = slot % 32;
        std::cout << "  m=" << m << " → offset=" << offset << " → byte=" << byte_offset 
                  << " → slot=" << slot << " → bank=" << bank << "\n";
    }
    
    return 0;
}
