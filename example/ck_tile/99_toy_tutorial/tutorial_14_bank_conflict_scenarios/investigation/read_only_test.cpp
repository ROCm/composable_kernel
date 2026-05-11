// Read-only LDS test - isolate read conflicts (transpose pattern)
// Expected WITHOUT XOR: 4,096 conflicts
// Expected WITH XOR: 0 conflicts
#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct ReadOnlyKernel
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

    // Transposed [K, M] LDS descriptor
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
                make_tuple(kK, kM),
                make_tuple(number<1>{}, number<kK>{}));
        }
    }

    CK_TILE_DEVICE void operator()(const DataType* __restrict__ input,
                                    DataType* __restrict__ output,
                                    index_t M,
                                    index_t K) const
    {
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        if(block_m >= M) return;

        // Initialize LDS with input data (use simple direct write to minimize write conflicts)
        // This is a setup phase - we want to isolate READ conflicts
        for(index_t i = threadIdx.x; i < kM * kK; i += kBlockSize)
        {
            lds[i] = input[block_m * K + i % kK + (i / kK) * K];
        }
        block_sync_lds();

        // Loop over K dimension
        for(index_t k_block = 0; k_block < K; k_block += kK)
        {
            // TRANSPOSE READ: Create [K, M] view of LDS buffer
            constexpr auto lds_desc_km = MakeLdsDescriptorKM();

            auto lds_view_km = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<DataType*>(lds), lds_desc_km);

            // Distribution for [K, M] - same as production kernel
            constexpr index_t M1 = 16 / sizeof(DataType);
            constexpr index_t M0 = kM / M1;
            constexpr index_t K2 = 64 / M0;
            constexpr index_t K1 = kBlockSize / 64;
            constexpr index_t K0 = kK / (K2 * K1);

            constexpr auto dist_km = make_static_tile_distribution(
                tile_distribution_encoding<
                    sequence<1>,
                    tuple<sequence<K0, K1, K2>, sequence<M0, M1>>,
                    tuple<sequence<1>, sequence<1, 2>>,
                    tuple<sequence<1>, sequence<2, 0>>,
                    sequence<1, 2>,
                    sequence<0, 1>
                >{});

            auto lds_window_km = make_tile_window(
                lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

            // THIS IS WHAT WE'RE MEASURING - transpose read from LDS
            auto reg_final = load_tile(lds_window_km);

            block_sync_lds();

            // Write to global to prevent optimization
            const auto gmem_desc_out = make_naive_tensor_descriptor(
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(M, number<1>{}));

            auto gmem_view_out = make_tensor_view<address_space_enum::global>(
                output + k_block * M + block_m, gmem_desc_out);

            auto gmem_window_out = make_tile_window(
                gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km);

            store_tile(gmem_window_out, reg_final);

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
    std::vector<DataType> h_output(K * M);

    for(index_t i = 0; i < M * K; ++i)
        h_input[i] = static_cast<DataType>(i);

    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(K * M * sizeof(DataType));

    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    constexpr index_t kM = 64;
    constexpr index_t block_size = 256;
    const index_t grid_size = (M + kM - 1) / kM;

    std::cout << "  XOR: " << (UseXor ? "ENABLED" : "DISABLED") << "\n";

    stream_config stream;
    constexpr index_t lds_size = ReadOnlyKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     ReadOnlyKernel<DataType, UseXor>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());

    std::cout << "  Read-only kernel completed\n";
    return true;
}

int main()
{
    std::cout << "\n========================================\n";
    std::cout << "READ-ONLY LDS Test (Transpose Pattern)\n";
    std::cout << "Expected WITHOUT XOR: 4,096 conflicts\n";
    std::cout << "Expected WITH XOR: 0 conflicts\n";
    std::cout << "========================================\n";

    run_test<false>("Test 1: Read-only WITHOUT XOR");
    run_test<true>("Test 2: Read-only WITH XOR");

    std::cout << "\nExpected conflicts (read only):\n";
    std::cout << "  WITHOUT XOR:\n";
    std::cout << "    Intra-lane: 4 slots/bank × 2 banks = 4 conflicts per lane\n";
    std::cout << "    8 lanes × 4 k_bases × 4 = 128 per phase\n";
    std::cout << "    8 phases × 4 K-iters × 128 = 4,096 total\n";
    std::cout << "  WITH XOR:\n";
    std::cout << "    All 8 slots hit different banks = 0 conflicts\n";

    return 0;
}
