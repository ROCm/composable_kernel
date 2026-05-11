// Debug version: Print bank information from the actual descriptor during kernel execution
#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct DebugBanksKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // LDS descriptor for [M, K] - WRITE
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

    // LDS descriptor for [K, M] - READ (transposed)
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

    // Row-major [M, K] distribution for WRITE
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
                                    int* __restrict__ bank_info,  // Output: bank info per thread
                                    index_t M,
                                    index_t K) const
    {
        __shared__ DataType lds[kM * kK];

        const index_t block_m = blockIdx.x * kM;
        if(block_m >= M) return;

        const index_t tid = threadIdx.x;

        // Process all 256 threads (4 wavefronts)

        // Setup LDS descriptors
        constexpr auto lds_desc_mk = MakeLdsDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_mk);

        constexpr auto dist_mk = MakeDistributionMK();

        auto lds_window_mk = make_tile_window(
            lds_view_mk, make_tuple(kM, kK), {0, 0}, dist_mk);

        // Global input
        const auto gmem_desc_in = make_naive_tensor_descriptor(
            make_tuple(number<kM>{}, number<kK>{}),
            make_tuple(K, number<1>{}));

        auto gmem_view_in = make_tensor_view<address_space_enum::global>(
            input + block_m * K, gmem_desc_in);

        auto gmem_window_in = make_tile_window(
            gmem_view_in, make_tuple(kM, kK), {0, 0}, dist_mk);

        // Load and store to LDS
        auto reg_tile = load_tile(gmem_window_in);
        store_tile(lds_window_mk, reg_tile);

        block_sync_lds();

        // Now setup the READ (transpose) descriptor
        constexpr auto lds_desc_km = MakeLdsDescriptorKM();

        // Distribution for [K, M]
        constexpr index_t M1_read = 16 / sizeof(DataType);
        constexpr index_t M0_read = kM / M1_read;
        constexpr index_t K2 = 64 / M0_read;
        constexpr index_t K1 = kBlockSize / 64;
        constexpr index_t K0 = kK / (K2 * K1);

        constexpr auto dist_km = make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<K0, K1, K2>, sequence<M0_read, M1_read>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>
            >{});

        // Calculate bank info for this thread
        // Thread tid accesses: K2_idx = tid % 8, M0_idx = tid / 8
        // For K1 = 0..3 iterations, for M1 = 0..7 elements

        constexpr index_t DataTypeSize = sizeof(DataType);

        // All 256 threads participate
        // K1 = tid / 64 (which wavefront: 0-3)
        // Within wavefront: k2_idx = (tid % 64) % 8, m0_idx = (tid % 64) / 8
        index_t k1_idx = tid / 64;
        index_t lane_in_wf = tid % 64;
        index_t k2_idx = lane_in_wf % 8;
        index_t m0_idx = lane_in_wf / 8;

        // Store bank pattern for this thread's 8 M elements
        for(index_t m1 = 0; m1 < 8; m1++)
        {
            index_t k = k1_idx * 8 + k2_idx;
            index_t m = m0_idx * 8 + m1;

            // Calculate offset using the descriptor
            auto offset = lds_desc_km.calculate_offset(make_tuple(k, m));
            index_t byte_offset = offset * DataTypeSize;
            index_t slot = byte_offset / 4;
            index_t bank = slot % 32;

            // Store: bank_info[tid * 8 + m1] = bank
            bank_info[tid * 8 + m1] = static_cast<int>(bank);
        }

        // Also do the actual load for verification
        auto lds_view_km = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_km);

        auto lds_window_km = make_tile_window(
            lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

        auto reg_final = load_tile(lds_window_km);

        block_sync_lds();

        // Store transposed output
        const auto gmem_desc_out = make_naive_tensor_descriptor(
            make_tuple(number<kK>{}, number<kM>{}),
            make_tuple(M, number<1>{}));

        auto gmem_view_out = make_tensor_view<address_space_enum::global>(
            output + block_m, gmem_desc_out);

        auto gmem_window_out = make_tile_window(
            gmem_view_out, make_tuple(kK, kM), {0, 0}, dist_km);

        store_tile(gmem_window_out, reg_final);
    }
};

template<bool UseXor>
void run_and_print_banks(const std::string& name)
{
    std::cout << "\n========================================\n";
    std::cout << name << "\n";
    std::cout << "========================================\n\n";

    constexpr index_t M = 64;  // Just one block for debugging
    constexpr index_t K = 32;

    using DataType = half_t;

    std::vector<DataType> h_input(M * K);
    std::vector<DataType> h_output(K * M);
    std::vector<int> h_bank_info(256 * 8);  // 256 threads × 8 M elements

    for(index_t m = 0; m < M; ++m)
        for(index_t k = 0; k < K; ++k)
            h_input[m * K + k] = static_cast<DataType>(m * 1000 + k);

    DeviceMem d_input(M * K * sizeof(DataType));
    DeviceMem d_output(K * M * sizeof(DataType));
    DeviceMem d_bank_info(256 * 8 * sizeof(int));

    d_input.ToDevice(h_input.data(), M * K * sizeof(DataType));

    constexpr index_t block_size = 256;
    const index_t grid_size = 1;

    stream_config stream;
    constexpr index_t lds_size = DebugBanksKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     DebugBanksKernel<DataType, UseXor>{},
                     dim3(grid_size),
                     dim3(block_size),
                     lds_size,
                     static_cast<const DataType*>(d_input.GetDeviceBuffer()),
                     static_cast<DataType*>(d_output.GetDeviceBuffer()),
                     static_cast<int*>(d_bank_info.GetDeviceBuffer()),
                     M, K));

    hip_check_error(hipDeviceSynchronize());

    d_bank_info.FromDevice(h_bank_info.data(), 256 * 8 * sizeof(int));

    // Print bank info for each wavefront
    for(int wf = 0; wf < 4; wf++)
    {
        std::cout << "\n--- Wavefront " << wf << " (threads " << wf*64 << "-" << (wf*64+63) << ", k=" << wf*8 << "-" << (wf*8+7) << ") ---\n";
        std::cout << "Thread | K1 | K2 | M0 | Banks for m1=0..7\n";
        std::cout << "-------|----|----|----|-----------------\n";

        for(int lane = 0; lane < 64; lane += 8)  // Print every 8th thread
        {
            int tid = wf * 64 + lane;
            int k1_idx = wf;
            int k2_idx = lane % 8;
            int m0_idx = lane / 8;

            printf("  %3d  | %2d | %2d | %2d | ", tid, k1_idx, k2_idx, m0_idx);
            for(int m1 = 0; m1 < 8; m1++)
            {
                printf("%2d ", h_bank_info[tid * 8 + m1]);
            }
            printf("\n");
        }
    }

    // Count conflicts for m1=0 across ALL 256 threads
    std::cout << "\n=== Conflict Analysis for m1=0 (ALL 256 threads) ===\n";

    int bank_counts[32] = {0};
    for(int tid = 0; tid < 256; tid++)
    {
        int bank = h_bank_info[tid * 8 + 0];  // m1=0
        bank_counts[bank]++;
    }

    int total_conflicts = 0;
    for(int bank = 0; bank < 32; bank++)
    {
        if(bank_counts[bank] > 1)
        {
            std::cout << "Bank " << bank << ": " << bank_counts[bank] << " threads → "
                      << (bank_counts[bank] - 1) << " conflicts\n";
            total_conflicts += (bank_counts[bank] - 1);
        }
    }
    std::cout << "\nTotal conflicts for m1=0 (256 threads): " << total_conflicts << "\n";
    std::cout << "Estimated total (×8 dm steps): " << total_conflicts * 8 << "\n";
    std::cout << "Scaled (×4 blocks): " << total_conflicts * 8 * 4 << "\n";
}

int main()
{
    std::cout << "\n╔═══════════════════════════════════════════════════╗\n";
    std::cout << "║ Debug Banks - Print actual bank access pattern    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════╝\n";

    run_and_print_banks<false>("WITHOUT XOR");
    run_and_print_banks<true>("WITH XOR");

    return 0;
}
