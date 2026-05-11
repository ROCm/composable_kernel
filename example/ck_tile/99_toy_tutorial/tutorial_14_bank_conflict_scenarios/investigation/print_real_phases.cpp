// Print actual phase groupings using the real KM distribution

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;
using DataType = half_t;

constexpr index_t kM = 64;
constexpr index_t kK = 32;
constexpr index_t kKPack = 8;
constexpr index_t kBlockSize = 256;

// KM distribution for transpose read (from 01_row_major.cpp)
CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionKM()
{
    constexpr index_t M1 = 16 / sizeof(DataType);  // 8
    constexpr index_t M0 = kM / M1;                 // 8
    constexpr index_t K2 = 64 / M0;                 // 8
    constexpr index_t K1 = kBlockSize / 64;         // 4
    constexpr index_t K0 = kK / (K2 * K1);          // 1

    return make_static_tile_distribution(
        tile_distribution_encoding<
            sequence<1>,
            tuple<sequence<K0, K1, K2>, sequence<M0, M1>>,
            tuple<sequence<1>, sequence<1, 2>>,
            tuple<sequence<1>, sequence<2, 0>>,
            sequence<1, 2>,
            sequence<0, 1>
        >{});
}

// XOR descriptor
CK_TILE_HOST_DEVICE static constexpr auto MakeXorDescriptor()
{
    constexpr auto DataTypeSize = sizeof(DataType);
    constexpr auto MLdsLayer = (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);

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
            make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<MLdsLayer>{})),
            make_merge_transform(make_tuple(number<kK / kKPack>{}, number<kKPack>{}))),
        make_tuple(sequence<1, 0>{}, sequence<2, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return lds_desc;
}

// Kernel to print thread mappings
__global__ void print_thread_mapping_kernel(int* k_out, int* m_out, int* bank_out)
{
    constexpr auto xor_desc = MakeXorDescriptor();
    constexpr auto dist_km = MakeDistributionKM();

    int tid = threadIdx.x;

    // Get the thread's slice in the distribution
    // The distribution tells us which (k, m) coordinates this thread handles

    // For now, let's just compute based on the distribution encoding
    // K0=1, K1=4, K2=8 -> K dimension
    // M0=8, M1=8 -> M dimension
    // Thread mapping from encoding:
    //   tuple<sequence<1>, sequence<1, 2>>  -> replicate dims
    //   tuple<sequence<1>, sequence<2, 0>>  -> partition dims
    // This means:
    //   K1 (4) is partitioned by dim 1 (wavefront index)
    //   K2 (8) is partitioned by dim 2 (lane within wavefront)
    //   M0 (8) is partitioned by dim 2 (same as K2)
    //   M1 (8) is NOT partitioned - each thread handles all 8 M1 values

    int wf = tid / 64;          // wavefront index (0-3)
    int lane = tid % 64;        // lane within wavefront (0-63)

    // From the distribution:
    // K1 index = wf (partitioned by dim 1)
    // K2 index = lane / 8 (partitioned by dim 2, with M0)
    // M0 index = lane % 8 (partitioned by dim 2)
    // M1 is the "repeat" dimension - each thread reads 8 M1 values

    int k1 = wf;
    int k2 = lane / 8;
    int m0 = lane % 8;

    // K = K0 * K1 * K2 + ... but K0=1, so:
    // K = k1 * K2 + k2 = wf * 8 + lane/8
    int k = k1 * 8 + k2;

    // For M, each thread handles M1=8 values
    // Base M = m0 * M1 = (lane % 8) * 8
    int m_base = m0 * 8;

    // Store the k and m_base for this thread
    k_out[tid] = k;
    m_out[tid] = m_base;

    // Calculate bank for first m value (dm=0)
    int m = m_base + 0;
    auto offset = xor_desc.calculate_offset(make_multi_index(m, k));
    int byte_addr = offset * sizeof(DataType);
    int bank = (byte_addr / 4) % 32;
    bank_out[tid] = bank;
}

int main()
{
    std::cout << "=== Real Phase Groupings for KM Distribution ===\n\n";

    // Print distribution info
    std::cout << "Distribution encoding:\n";
    std::cout << "  K: K0=1, K1=4 (wavefronts), K2=8 (lanes/8)\n";
    std::cout << "  M: M0=8 (lane%8), M1=8 (repeat per thread)\n";
    std::cout << "\n";

    std::vector<int> h_k(kBlockSize), h_m(kBlockSize), h_bank(kBlockSize);
    DeviceMem d_k(kBlockSize * sizeof(int));
    DeviceMem d_m(kBlockSize * sizeof(int));
    DeviceMem d_bank(kBlockSize * sizeof(int));

    print_thread_mapping_kernel<<<1, kBlockSize>>>(
        static_cast<int*>(d_k.GetDeviceBuffer()),
        static_cast<int*>(d_m.GetDeviceBuffer()),
        static_cast<int*>(d_bank.GetDeviceBuffer()));
    hip_check_error(hipDeviceSynchronize());

    d_k.FromDevice(h_k.data(), kBlockSize * sizeof(int));
    d_m.FromDevice(h_m.data(), kBlockSize * sizeof(int));
    d_bank.FromDevice(h_bank.data(), kBlockSize * sizeof(int));

    // Print for WF0 (first 64 threads)
    std::cout << "WF0 Thread Mapping (dm=0):\n";
    std::cout << "Lane | k | m_base | bank\n";
    std::cout << "-----|---|--------|-----\n";
    for (int lane = 0; lane < 64; lane++)
    {
        std::cout << lane << " | " << h_k[lane] << " | " << h_m[lane]
                  << " | " << h_bank[lane] << "\n";
    }

    // Group by phase (user says phase 0 = lanes 0-3 and 20-23)
    std::cout << "\n=== Phase groupings (user specified: 0-3, 20-23 etc) ===\n";

    // Let me group by which threads access simultaneously
    // On AMD, ds_read executes in phases based on SIMD structure
    // For gfx9, it's typically 4 phases of 16 threads or some other grouping

    // Let's try the user's grouping: phase 0 = {0-3, 20-23}
    // This suggests lanes 0-3 and lanes 20-23 execute together
    std::vector<std::vector<int>> user_phases = {
        {0, 1, 2, 3, 20, 21, 22, 23},
        {4, 5, 6, 7, 24, 25, 26, 27},
        {8, 9, 10, 11, 28, 29, 30, 31},
        {12, 13, 14, 15, 32, 33, 34, 35},
        {16, 17, 18, 19, 36, 37, 38, 39},
        // ... need to figure out rest
    };

    // Actually let me just print what banks lanes 0-3 and 20-23 hit
    std::cout << "\nPhase 0 (if lanes 0-3, 20-23):\n";
    std::cout << "Lane | k | m | bank\n";
    std::map<int, std::vector<int>> phase0_banks;
    for (int lane : {0, 1, 2, 3, 20, 21, 22, 23})
    {
        std::cout << lane << " | " << h_k[lane] << " | " << h_m[lane]
                  << " | " << h_bank[lane] << "\n";
        phase0_banks[h_bank[lane]].push_back(lane);
    }

    std::cout << "\nBanks: ";
    for (const auto& [bank, lanes] : phase0_banks)
    {
        std::cout << "B" << bank << "(" << lanes.size() << ") ";
    }
    std::cout << "\n";

    return 0;
}
