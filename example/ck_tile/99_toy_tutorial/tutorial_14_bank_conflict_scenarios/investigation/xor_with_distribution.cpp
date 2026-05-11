// Complete example: XOR descriptor + tile distribution + load_tile
//
// Shows the full flow:
// 1. Fill LDS with known pattern using XOR descriptor (store)
// 2. Create tile window with distribution for transpose read
// 3. Use load_tile to read into registers
// 4. Print which values each thread gets

#include <iostream>
#include <iomanip>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;
using DataType = half_t;

constexpr index_t kM = 64;
constexpr index_t kK = 32;
constexpr index_t kKPack = 8;
constexpr index_t kBlockSize = 256;

// XOR descriptor [M, K] for writing
CK_TILE_HOST_DEVICE static constexpr auto MakeXorDescriptorMK()
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

// XOR descriptor [K, M] for reading (transpose)
CK_TILE_HOST_DEVICE static constexpr auto MakeXorDescriptorKM()
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

    // Different merge order for transpose!
    constexpr auto lds_desc = transform_tensor_descriptor(
        lds_desc_unmerged,
        make_tuple(
            make_merge_transform(make_tuple(number<MLdsLayer>{}, number<kK / kKPack>{})),
            make_merge_transform(make_tuple(number<kM / MLdsLayer>{}, number<kKPack>{}))),
        make_tuple(sequence<0, 2>{}, sequence<1, 3>{}),
        make_tuple(sequence<0>{}, sequence<1>{}));

    return lds_desc;
}

// Distribution for [M, K] writing
CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionMK()
{
    constexpr index_t K1 = 16 / sizeof(DataType);  // 8 for FP16
    constexpr index_t K0 = kK / K1;                 // 4
    constexpr index_t M2 = 64 / K0;                 // 16
    constexpr index_t M1 = kBlockSize / 64;         // 4
    constexpr index_t M0 = kM / (M2 * M1);          // 1

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

// Distribution for [K, M] transpose read
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

struct XorDistributionKernel
{
    static constexpr index_t kBlockSize = 256;

    CK_TILE_DEVICE void operator()(int* output) const
    {
        __shared__ DataType lds[kM * kK];

        int tid = threadIdx.x;
        int wf = tid / 64;
        int lane = tid % 64;

        // Step 1: Fill LDS with pattern: value = m * 1000 + k
        // Write using XOR descriptor [M, K]
        constexpr auto xor_desc_mk = MakeXorDescriptorMK();
        auto lds_view_mk = make_tensor_view<address_space_enum::lds>(lds, xor_desc_mk);

        // Each thread fills some entries
        for (int i = tid; i < kM * kK; i += kBlockSize)
        {
            int m = i / kK;
            int k = i % kK;
            auto offset = xor_desc_mk.calculate_offset(make_multi_index(m, k));
            lds[offset] = static_cast<DataType>(m * 1000 + k);
        }
        __syncthreads();

        // Step 2: Read using XOR descriptor [K, M] with distribution
        constexpr auto xor_desc_km = MakeXorDescriptorKM();
        constexpr auto dist_km = MakeDistributionKM();

        auto lds_view_km = make_tensor_view<address_space_enum::lds>(lds, xor_desc_km);
        auto lds_window = make_tile_window(lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

        // Step 3: Load tile into registers
        auto reg_tile = load_tile(lds_window);

        __syncthreads();

        // Step 4: Each thread prints what it read
        if (tid < 64)  // Only first wavefront for now
        {
            auto thread_buf = reg_tile.get_thread_buffer();

            // Output format: tid, wf, lane, then 8 values
            // Each thread reads M1=8 values
            output[tid * 10 + 0] = tid;
            output[tid * 10 + 1] = wf;
            for (int i = 0; i < 8; i++)
            {
                output[tid * 10 + 2 + i] = static_cast<int>(thread_buf[i]);
            }
        }
    }
};

int main()
{
    std::cout << "=== XOR Descriptor + Distribution + Load Tile ===\n\n";

    std::cout << "Setup:\n";
    std::cout << "  Matrix: [" << kM << ", " << kK << "] stored as [M, K]\n";
    std::cout << "  Value at (m, k) = m * 1000 + k\n";
    std::cout << "  Write: XOR descriptor [M, K] + MK distribution\n";
    std::cout << "  Read:  XOR descriptor [K, M] + KM distribution\n\n";

    std::cout << "Two distributions (like actual transpose kernels):\n";
    std::cout << "  MK dist: M0=1, M1=4, M2=16; K0=4, K1=8\n";
    std::cout << "  KM dist: K0=1, K1=4, K2=8; M0=8, M1=8\n\n";

    // Launch kernel
    std::vector<int> h_output(64 * 10, -1);
    DeviceMem d_output(64 * 10 * sizeof(int));

    stream_config stream;
    launch_kernel(stream,
                  make_kernel<kBlockSize>(
                      XorDistributionKernel{},
                      dim3(1),
                      dim3(kBlockSize),
                      0,
                      static_cast<int*>(d_output.GetDeviceBuffer())));

    hip_check_error(hipDeviceSynchronize());
    d_output.FromDevice(h_output.data(), 64 * 10 * sizeof(int));

    std::cout << "Distribution analysis:\n";
    std::cout << "  K: K0=1, K1=4, K2=8 (k = wf*8 + lane/8)\n";
    std::cout << "  M: M0=8, M1=8 (m_base = (lane%8)*8, each thread reads 8 M values)\n\n";

    std::cout << "Thread reads (WF0, first 8 threads):\n";
    std::cout << "Tid | WF | Lane | Values [8] (format: m*1000+k)\n";
    std::cout << "----|----|----- |------------------------------------------\n";

    for (int tid = 0; tid < 8; tid++)
    {
        int wf = h_output[tid * 10 + 1];
        std::cout << std::setw(3) << tid << " | "
                  << wf << "  | "
                  << std::setw(4) << tid << " | [";

        for (int i = 0; i < 8; i++)
        {
            std::cout << std::setw(5) << h_output[tid * 10 + 2 + i];
            if (i < 7) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    std::cout << "\nDecoding thread 0's values:\n";
    for (int i = 0; i < 8; i++)
    {
        int val = h_output[0 * 10 + 2 + i];
        int m = val / 1000;
        int k = val % 1000;
        std::cout << "  val[" << i << "] = " << val << " = m=" << m << ", k=" << k << "\n";
    }

    // Show expected pattern based on distribution
    std::cout << "\n=== Expected Pattern (from distribution) ===\n";
    std::cout << "Lane 0: k=0/64=0, m_base=(0%8)*8=0, reads m=0-7 at k=0\n";
    std::cout << "        Values: [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000]\n\n";

    std::cout << "Lane 1: k=0, m_base=8, reads m=8-15 at k=0\n";
    std::cout << "        Values: [8000, 9000, 10000, ...]\n\n";

    std::cout << "Lane 8: k=1, m_base=0, reads m=0-7 at k=1\n";
    std::cout << "        Values: [1, 1001, 2001, 3001, ...]\n\n";

    // Verify against expected
    bool match = true;
    for (int lane = 0; lane < 8; lane++)
    {
        int k = lane / 8;  // 0 for all in this phase
        int m_base = (lane % 8) * 8;

        for (int dm = 0; dm < 8; dm++)
        {
            int expected = (m_base + dm) * 1000 + k;
            int actual = h_output[lane * 10 + 2 + dm];
            if (expected != actual)
            {
                std::cout << "MISMATCH at lane " << lane << ", dm=" << dm
                          << ": expected " << expected << ", got " << actual << "\n";
                match = false;
            }
        }
    }

    if (match)
    {
        std::cout << "✓ All values match expected pattern!\n";
    }

    std::cout << "\n=== Bank Analysis ===\n";
    std::cout << "For transpose read [K, M], lanes in phase 0 access:\n";
    std::cout << "  Lane 0-7 all read k=0 with different m values\n";
    std::cout << "  This is a column read from the original [M, K] matrix\n\n";

    // Show which banks phase 0 hits
    constexpr auto xor_desc_mk = MakeXorDescriptorMK();
    std::cout << "Banks accessed by phase 0 (lanes 0-7, dm=0):\n";
    std::cout << "Lane | k | m | offset | bank\n";
    std::cout << "-----|---|---|--------|-----\n";

    for (int lane = 0; lane < 8; lane++)
    {
        int k = 0;
        int m = lane * 8;
        auto offset = xor_desc_mk.calculate_offset(make_multi_index(m, k));
        int byte_addr = offset * sizeof(DataType);
        int bank = (byte_addr / 4) % 32;

        std::cout << "  " << lane << "  | " << k << " | " << std::setw(2) << m
                  << " | " << std::setw(6) << offset << " | " << std::setw(4) << bank << "\n";
    }

    return 0;
}
