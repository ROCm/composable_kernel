// Debug version: Extract bank info directly from tile window's precomputed coordinates
// This shows how the tile window internally calculates offsets
#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType, bool UseXor>
struct DebugBanksFromWindowKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;
    static constexpr index_t kKPack = 8;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
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
                make_tuple(number<kK>{}, number<kM>{}),
                make_tuple(number<1>{}, number<kK>{}));
        }
    }

    // Distribution for [K, M] - READ
    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionKM()
    {
        constexpr index_t M1_read = 16 / sizeof(DataType);
        constexpr index_t M0_read = kM / M1_read;
        constexpr index_t K2 = 64 / M0_read;
        constexpr index_t K1 = kBlockSize / 64;
        constexpr index_t K0 = kK / (K2 * K1);

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<K0, K1, K2>, sequence<M0_read, M1_read>>,
                tuple<sequence<1>, sequence<1, 2>>,
                tuple<sequence<1>, sequence<2, 0>>,
                sequence<1, 2>,
                sequence<0, 1>
            >{});
    }

    CK_TILE_DEVICE void operator()(int* __restrict__ bank_info,  // Output: bank info per thread
                                    int* __restrict__ offset_info) const  // Output: raw offsets
    {
        __shared__ DataType lds[kM * kK];

        const index_t tid = threadIdx.x;

        // Setup LDS view and window
        constexpr auto lds_desc_km = MakeLdsDescriptorKM();
        auto lds_view_km = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<DataType*>(lds), lds_desc_km);

        constexpr auto dist_km = MakeDistributionKM();
        auto lds_window_km = make_tile_window(
            lds_view_km, make_tuple(number<kK>{}, number<kM>{}), {0, 0}, dist_km);

        // The tile window internally has pre_computed_coords_ which contains the offsets
        // We can access the bottom tensor view's descriptor to calculate offsets

        // Get the descriptor from the view
        const auto& desc = lds_view_km.get_tensor_descriptor();

        constexpr index_t DataTypeSize = sizeof(DataType);

        // Calculate which (k, m) values this thread accesses
        // Based on distribution: K1 = tid / 64, within wavefront: k2 = (tid%64)%8, m0 = (tid%64)/8
        index_t k1_idx = tid / 64;
        index_t lane_in_wf = tid % 64;
        index_t k2_idx = lane_in_wf % 8;
        index_t m0_idx = lane_in_wf / 8;

        // Store bank pattern for this thread's 8 M elements (M1 = 8)
        for(index_t m1 = 0; m1 < 8; m1++)
        {
            index_t k = k1_idx * 8 + k2_idx;  // Full k coordinate
            index_t m = m0_idx * 8 + m1;      // Full m coordinate

            // Calculate offset using the descriptor (same as tile window does internally)
            auto offset = desc.calculate_offset(make_tuple(k, m));
            index_t byte_offset = offset * DataTypeSize;
            index_t slot = byte_offset / 4;
            index_t bank = slot % 32;

            // Store results
            bank_info[tid * 8 + m1] = static_cast<int>(bank);
            offset_info[tid * 8 + m1] = static_cast<int>(offset);
        }
    }
};

template<bool UseXor>
void run_and_print_banks(const std::string& name)
{
    std::cout << "\n========================================\n";
    std::cout << name << "\n";
    std::cout << "========================================\n\n";

    using DataType = half_t;

    std::vector<int> h_bank_info(256 * 8);
    std::vector<int> h_offset_info(256 * 8);

    DeviceMem d_bank_info(256 * 8 * sizeof(int));
    DeviceMem d_offset_info(256 * 8 * sizeof(int));

    constexpr index_t block_size = 256;

    stream_config stream;
    constexpr index_t lds_size = DebugBanksFromWindowKernel<DataType, UseXor>::GetStaticLdsSize();

    launch_kernel(stream,
                 make_kernel<block_size>(
                     DebugBanksFromWindowKernel<DataType, UseXor>{},
                     dim3(1),
                     dim3(block_size),
                     lds_size,
                     static_cast<int*>(d_bank_info.GetDeviceBuffer()),
                     static_cast<int*>(d_offset_info.GetDeviceBuffer())));

    hip_check_error(hipDeviceSynchronize());

    d_bank_info.FromDevice(h_bank_info.data(), 256 * 8 * sizeof(int));
    d_offset_info.FromDevice(h_offset_info.data(), 256 * 8 * sizeof(int));

    // Print detailed bank info for first wavefront (threads 0-63)
    std::cout << "First wavefront (threads 0-63), K1=0, k=0..7:\n\n";
    std::cout << "Lane | K2 | M0 | Offsets for m1=0..7 | Banks for m1=0..7\n";
    std::cout << "-----|----|----|---------------------|-------------------\n";

    for(int lane = 0; lane < 64; lane += 8)  // Print every 8th thread for brevity
    {
        int k2_idx = lane % 8;
        int m0_idx = lane / 8;

        printf(" %2d  | %2d | %2d | ", lane, k2_idx, m0_idx);
        for(int m1 = 0; m1 < 8; m1++)
        {
            printf("%4d ", h_offset_info[lane * 8 + m1]);
        }
        printf("| ");
        for(int m1 = 0; m1 < 8; m1++)
        {
            printf("%2d ", h_bank_info[lane * 8 + m1]);
        }
        printf("\n");
    }

    // Slot-based conflict analysis
    std::cout << "\n=== SLOT-BASED CONFLICT ANALYSIS ===\n";
    std::cout << "Rule: Multiple threads accessing SAME slot = 0 conflicts\n";
    std::cout << "      (unique_slots_per_bank - 1) = conflicts per bank\n\n";

    index_t total_conflicts = 0;

    for(int k1 = 0; k1 < 4; k1++)
    {
        index_t k1_conflicts = 0;

        for(int dm = 0; dm < 8; dm++)
        {
            // Count unique slots per bank
            int bank_slots[32][64] = {0};  // bank -> array of slots
            int bank_slot_count[32] = {0};

            for(int lane = 0; lane < 64; lane++)
            {
                int tid = k1 * 64 + lane;
                int offset = h_offset_info[tid * 8 + dm];
                int byte_offset = offset * 2;  // FP16
                int slot = byte_offset / 4;
                int bank = slot % 32;

                // Check if slot already counted for this bank
                bool found = false;
                for(int i = 0; i < bank_slot_count[bank]; i++)
                {
                    if(bank_slots[bank][i] == slot)
                    {
                        found = true;
                        break;
                    }
                }
                if(!found)
                {
                    bank_slots[bank][bank_slot_count[bank]++] = slot;
                }
            }

            // Conflicts = unique_slots - 1 for each bank
            for(int bank = 0; bank < 32; bank++)
            {
                if(bank_slot_count[bank] > 1)
                {
                    k1_conflicts += (bank_slot_count[bank] - 1);
                }
            }
        }

        std::cout << "K1=" << k1 << " (WF " << k1 << "): " << k1_conflicts << " conflicts\n";
        total_conflicts += k1_conflicts;
    }

    std::cout << "\nTotal per tile: " << total_conflicts << "\n";
    std::cout << "Scaled (4 blocks): " << total_conflicts * 4 << "\n";
}

int main()
{
    std::cout << "\n╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║ Debug Banks from Tile Window - Using Real Descriptors  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n";

    run_and_print_banks<false>("WITHOUT XOR");
    run_and_print_banks<true>("WITH XOR");

    std::cout << "\n========================================\n";
    std::cout << "PROFILER COMPARISON:\n";
    std::cout << "  WITHOUT XOR: 7,168\n";
    std::cout << "  WITH XOR:    3,072\n";
    std::cout << "========================================\n";

    return 0;
}
