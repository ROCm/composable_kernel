// SPDX-License-Identifier: MIT
// Debug version of 01_row_major.cpp to understand thread-to-coordinate mapping

#include <iostream>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;

template<typename DataType>
struct DebugRowMajorKernel
{
    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kM = 64;
    static constexpr index_t kK = 32;

    CK_TILE_HOST_DEVICE static constexpr index_t GetStaticLdsSize()
    {
        return kM * kK * sizeof(DataType);
    }

    // Plain row-major LDS descriptor [M, K]
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorMK()
    {
        return make_naive_tensor_descriptor_packed(make_tuple(number<kM>{}, number<kK>{}));
    }

    // Transposed view [K, M] for reading columns
    CK_TILE_HOST_DEVICE static constexpr auto MakeLdsDescriptorKM()
    {
        return make_naive_tensor_descriptor(
            make_tuple(number<kK>{}, number<kM>{}),
            make_tuple(number<1>{}, number<kK>{}));
    }

    // Distribution for [M, K] - for writing
    CK_TILE_HOST_DEVICE static constexpr auto MakeDistributionMK()
    {
        constexpr index_t K1 = 16 / sizeof(DataType);  // 8
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

    // Distribution for [K, M] - for reading (transpose)
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

    CK_TILE_DEVICE void operator()(DataType* __restrict__ output) const
    {
        __shared__ DataType lds[kM * kK];

        const int tid = threadIdx.x;
        const int wf = tid / 64;
        const int lane = tid % 64;

        // Initialize LDS with known pattern: value = m * 1000 + k
        for(int i = tid; i < kM * kK; i += kBlockSize)
        {
            int m = i / kK;
            int k = i % kK;
            lds[i] = static_cast<DataType>(m * 1000 + k);
        }
        __syncthreads();

        // Setup LDS view and window for KM distribution
        constexpr auto lds_desc_km = MakeLdsDescriptorKM();
        auto lds_view_km = make_tensor_view<address_space_enum::lds>(lds, lds_desc_km);
        constexpr auto dist_km = MakeDistributionKM();
        auto lds_window_km = make_tile_window(lds_view_km, make_tuple(kK, kM), {0, 0}, dist_km);

        // Load from LDS
        auto reg_tile = load_tile(lds_window_km);

        // Debug: Print what thread 0 of each wavefront reads
        __syncthreads();

        // Print first element of each thread's register tile
        // reg_tile is distributed - use get_x_ys to decode the coordinate
        if(tid < 64)
        {
            // For debugging, print the values read by first wavefront
            // The actual coordinates depend on distribution
            auto thread_buf = reg_tile.get_thread_buffer();

            // Print first value read by this lane
            printf("Lane %2d (WF%d): val[0]=%5d, val[1]=%5d, val[2]=%5d, val[3]=%5d, val[4]=%5d, val[5]=%5d, val[6]=%5d, val[7]=%5d\n",
                   lane, wf,
                   static_cast<int>(thread_buf[number<0>{}]),
                   static_cast<int>(thread_buf[number<1>{}]),
                   static_cast<int>(thread_buf[number<2>{}]),
                   static_cast<int>(thread_buf[number<3>{}]),
                   static_cast<int>(thread_buf[number<4>{}]),
                   static_cast<int>(thread_buf[number<5>{}]),
                   static_cast<int>(thread_buf[number<6>{}]),
                   static_cast<int>(thread_buf[number<7>{}]));
        }

        // Also output the first few values to global memory for verification
        if(tid < 64)
        {
            auto thread_buf = reg_tile.get_thread_buffer();
            for(int i = 0; i < 8; i++)
            {
                output[tid * 8 + i] = thread_buf[i];
            }
        }
    }
};

int main()
{
    std::cout << "Debug: Row-Major LDS Thread-to-Coordinate Mapping\n";
    std::cout << "==================================================\n\n";

    using DataType = half_t;

    // Output buffer for verification
    std::vector<DataType> h_output(64 * 8, static_cast<DataType>(0));
    DeviceMem d_output(64 * 8 * sizeof(DataType));

    constexpr index_t block_size = 256;
    constexpr index_t lds_size = DebugRowMajorKernel<DataType>::GetStaticLdsSize();

    stream_config stream;

    launch_kernel(stream,
                 make_kernel<block_size>(
                     DebugRowMajorKernel<DataType>{},
                     dim3(1),
                     dim3(block_size),
                     lds_size,
                     static_cast<DataType*>(d_output.GetDeviceBuffer())));

    hip_check_error(hipDeviceSynchronize());

    d_output.FromDevice(h_output.data(), 64 * 8 * sizeof(DataType));

    std::cout << "\nVerification from global memory:\n";
    std::cout << "Lane 0: [";
    for(int i = 0; i < 8; i++)
        std::cout << static_cast<int>(h_output[i]) << (i < 7 ? ", " : "");
    std::cout << "]\n";

    std::cout << "Lane 1: [";
    for(int i = 0; i < 8; i++)
        std::cout << static_cast<int>(h_output[8 + i]) << (i < 7 ? ", " : "");
    std::cout << "]\n";

    std::cout << "Lane 8: [";
    for(int i = 0; i < 8; i++)
        std::cout << static_cast<int>(h_output[64 + i]) << (i < 7 ? ", " : "");
    std::cout << "]\n";

    std::cout << "\nTo decode: value = m * 1000 + k\n";
    std::cout << "So value 0 = m=0, k=0\n";
    std::cout << "   value 1 = m=0, k=1\n";
    std::cout << "   value 1000 = m=1, k=0\n";

    return 0;
}
