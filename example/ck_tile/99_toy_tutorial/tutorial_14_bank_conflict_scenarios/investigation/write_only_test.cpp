// Write-only LDS test - isolate write conflicts
// Expected: 3,072 conflicts (FP16 pairing ×3)
#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct WriteOnlyKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // LDS descriptor for [M, K] - same as production kernel
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
                    make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
                    make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
                make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return lds_desc;
        }
        else
        {
            return make_naive_tensor_descriptor_packed(make_tuple(kM, kK));
        }
    }

    // Row-major [M, K] distribution
    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);
        constexpr index_t K0 = kK / K1;
        constexpr index_t M2 = 64 / K0;
        constexpr index_t M1 = kBlockSize / 64;
        constexpr index_t M0 = kM / (M2 * M1);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>
            >{});
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                    DataType* __restrict__ output,
                                    index_t M,
                                    index_t K) const
    {
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        if(block_m >= M) return;

        // Setup LDS descriptor for [M, K]
        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_mk);

        constexpr auto dist_mk = MakeDistributionMK();

        auto lds_window_mk = make_tile_window(
            lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);

        // Loop over K dimension
        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            // Global input descriptor
            const auto gmem_desc_in = make_naive_tensor_descriptor(
                make_tuple(number<kM>{}, number<kK>{}),
                make_tuple(K, number<1>{}));

            auto gmem_view_in = make_tensor_view<address_space_enum::global>(
                input + block_m * K + k_block, gmem_desc_in);

            auto gmem_window_in = make_tile_window(
                gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk);

            // Load from global and store to LDS - THIS IS WHAT WE'RE MEASURING
            auto reg_tile = load_tile(gmem_window_in);
            store_tile(lds_window_mk, reg_tile);

            block_sync_lds();

            // NO READ FROM LDS - just sync and continue
            // Write dummy output to prevent compiler optimization
            if(threadIdx.x == 0 && k_block == 0)
            {
                output[block_m] = lds[0];
            }

            block_sync_lds();
        }
    }
};

template<bool UseXor>
bool run_test(const std::string& test_name)
{
    std::cout << "\n" << test_name << "\n";

    constexpr index_t M = 256;
    constexpr index_t K = 128;

    using DataType = half_t;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(M);

    for(index_t i = 0; i < M * K; ++i)
        h_input[i] = static_cast<DataType>(i);

    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(M * sizeof(DataType));

    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    constexpr index_t kM = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kM - 1) / kM;

    std::cout << "  XOR: " << (UseXor ? "ENABLED" : "DISABLED") << "\n";

    stream_config stream;
    constexpr index_t lds_size = WriteOnlyKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     WriteOnlyKernel<DataType, UseXor>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());

    std::cout << "  Write-only kernel completed\n";
    return true;
}

int main()
{
    std::cout << "\n========================================\n";
    std::cout << "WRITE-ONLY LDS Test\n";
    std::cout << "Expected: 3,072 conflicts (both with/without XOR)\n";
    std::cout << "========================================\n";

    run_test<false>("Test 1: Write-only WITHOUT XOR");
    run_test<true>("Test 2: Write-only WITH XOR");

    std::cout << "\nExpected conflicts (write only):\n";
    std::cout << "  FP16 pairing: 8 lanes × 4 pairs × 3 = 96 per phase\n";
    std::cout << "  8 phases × 96 = 768 per K-iteration\n";
    std::cout << "  4 K-iterations × 768 = 3,072 total\n";

    return 0;
}
