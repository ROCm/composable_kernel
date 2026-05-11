// Test: What happens if we change M1 in the distribution?
// This is a minimal test to see if changing M1 affects the offset pattern

#include "common.hpp"

// Try M1 = 4 instead of 8
template <typename DataType, bool UseXor>
struct TestTransposeKernelM1_4
{
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kBlockSize = 256;

    CK_TILE_HOST_DEVICE void operator()(const DataType* __restrict__ input,
                                         DataType* __restrict__ output,
                                         index_t M,
                                         index_t K) const
    {
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        const index_t k_block = 0;

        // Load from global to LDS (same as before)
        constexpr auto gmem_desc_in = make_naive_tensor_descriptor(
            make_tuple(number<kM>{}, number<kK>{}),
            make_tuple(K, number<1>{}));

        auto gmem_view_in = make_tensor_view<address_space_enum::global>(
            input + block_m * K + k_block, gmem_desc_in);

        constexpr auto lds_desc_mk = make_naive_tensor_descriptor(
            make_tuple(number<kM>{}, number<kK>{}),
            make_tuple(number<kK>{}, number<1>{}));

        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(
            lds, lds_desc_mk);

        // ... (same loading code as before)

        // TRANSPOSE READ with M1 = 4 instead of 8
        constexpr auto lds_desc_km = make_naive_tensor_descriptor(
            make_tuple(number<kK>{}, number<kM>{}),
            make_tuple(number<kM>{}, number<1>{}));

        auto lds_view_km = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_km);

        // CHANGED: M1 = 4 instead of 8
        constexpr index_t M1 = 4;  // Was: 16 / sizeof(DataType) = 8
        constexpr index_t M0 = kM / M1; // 64 / 4 = 16 (was 8)
        constexpr index_t K2 = 64 / M0; // 64 / 16 = 4 (was 8)
        constexpr index_t K1 = kBlockSize / 64; // 256 / 64 = 4
        constexpr index_t K0 = kK / (K2 * K1); // 32 / (4 * 4) = 2 (was 1)

        // TODO: Need to complete this to actually test...
        // But this shows the parameter changes
    }
};

// Instructions:
// 1. Copy full kernel implementation from 04_row_major_xor.cpp
// 2. Change only M1 = 4
// 3. Compile and profile
// 4. Check assembly for offset patterns
// 5. Compare SQ_LDS_BANK_CONFLICT values

int main()
{
    std::cout << "This is a template for testing different M1 values.\n";
    std::cout << "To use:\n";
    std::cout << "1. Complete the kernel implementation\n";
    std::cout << "2. Compile with: hipcc -O2 -save-temps ...\n";
    std::cout << "3. Profile with rocprofv3\n";
    std::cout << "4. Compare assembly and conflict counts\n";

    std::cout << "\nCurrent M1 = 8:  3,072 conflicts (38%)\n";
    std::cout << "Test M1 = 4:     ??? conflicts\n";
    std::cout << "Test M1 = 16:    ??? conflicts\n";

    return 0;
}
