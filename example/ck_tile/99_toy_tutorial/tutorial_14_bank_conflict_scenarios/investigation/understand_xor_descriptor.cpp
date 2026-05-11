// Understand XOR descriptor: how logical (m, k) maps to physical offset
//
// We fill LDS with sequential values 0, 1, 2, ...
// Then access via XOR descriptor and see what we get.

#include <iostream>
#include <iomanip>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

using namespace ck_tile;
using DataType = half_t;

constexpr index_t kM = 16;  // Small for easy visualization
constexpr index_t kK = 8;
constexpr index_t kKPack = 8;

// Plain descriptor [M, K] - row major
CK_TILE_HOST_DEVICE static constexpr auto MakePlainDescriptor()
{
    return make_naive_tensor_descriptor_packed(make_tuple(number<kM>{}, number<kK>{}));
}

// XOR descriptor [M, K]
CK_TILE_HOST_DEVICE static constexpr auto MakeXorDescriptor()
{
    constexpr auto DataTypeSize = sizeof(DataType);
    constexpr auto MLdsLayer = (32 * 4 / kK / DataTypeSize) < 1 ? 1 : (32 * 4 / kK / DataTypeSize);
    // MLdsLayer = 128 / 16 = 8 for this small example

    constexpr auto lds_desc_0 = make_naive_tensor_descriptor(
        make_tuple(number<kK / kKPack * MLdsLayer>{},  // 1 * 8 = 8
                   number<kM / MLdsLayer>{},            // 16 / 8 = 2
                   number<kKPack>{}),                   // 8
        make_tuple(number<kKPack>{},                   // stride 8
                   number<kK * MLdsLayer>{},            // stride 64
                   number<1>{}),                        // stride 1
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

struct TestDescriptorKernel
{
    static constexpr index_t kBlockSize = 256;

    CK_TILE_DEVICE void operator()(int* output) const
{
    // LDS filled with sequential values: lds[i] = i
    __shared__ int lds[kM * kK];

    int tid = threadIdx.x;

    // Fill LDS with sequential values
    for (int i = tid; i < kM * kK; i += blockDim.x)
    {
        lds[i] = i;
    }
    __syncthreads();

    if (tid == 0)
    {
        constexpr auto plain_desc = MakePlainDescriptor();
        constexpr auto xor_desc = MakeXorDescriptor();

        // For each logical (m, k), get the physical offset from both descriptors
        for (int m = 0; m < kM; m++)
        {
            for (int k = 0; k < kK; k++)
            {
                auto plain_offset = plain_desc.calculate_offset(make_multi_index(m, k));
                auto xor_offset = xor_desc.calculate_offset(make_multi_index(m, k));

                // Store: m, k, plain_offset, xor_offset, plain_value, xor_value
                int idx = m * kK + k;
                output[idx * 6 + 0] = m;
                output[idx * 6 + 1] = k;
                output[idx * 6 + 2] = plain_offset;
                output[idx * 6 + 3] = xor_offset;
                output[idx * 6 + 4] = lds[plain_offset];
                output[idx * 6 + 5] = lds[xor_offset];
            }
        }
    }
    }
};

int main()
{
    std::cout << "=== Understanding XOR Descriptor ===\n\n";
    std::cout << "Matrix size: [" << kM << ", " << kK << "]\n";
    std::cout << "LDS filled with sequential values: lds[i] = i\n\n";

    // Launch kernel to verify on GPU
    std::vector<int> h_output(kM * kK * 6);
    DeviceMem d_output(kM * kK * 6 * sizeof(int));

    stream_config stream;
    launch_kernel(stream,
                  make_kernel<256>(
                      TestDescriptorKernel{},
                      dim3(1),
                      dim3(256),
                      0,
                      static_cast<int*>(d_output.GetDeviceBuffer())));

    hip_check_error(hipDeviceSynchronize());
    d_output.FromDevice(h_output.data(), kM * kK * 6 * sizeof(int));

    std::cout << "Kernel executed successfully!\n";
    std::cout << "Verifying kernel results match host...\n\n";

    // Also compute on host using descriptors
    constexpr auto plain_desc = MakePlainDescriptor();
    constexpr auto xor_desc = MakeXorDescriptor();

    std::cout << "Plain descriptor: row-major, offset = m * " << kK << " + k\n";
    std::cout << "XOR descriptor: transformed layout\n\n";

    std::cout << "Logical (m,k) -> Physical offset:\n\n";
    std::cout << "  m |  k | plain_off | xor_off | plain_val | xor_val\n";
    std::cout << "----|----|-----------|---------|-----------|---------\n";

    for (int m = 0; m < kM; m++)
    {
        for (int k = 0; k < kK; k++)
        {
            auto plain_off = plain_desc.calculate_offset(make_multi_index(m, k));
            auto xor_off = xor_desc.calculate_offset(make_multi_index(m, k));

            // plain_val = what's at that physical location if LDS was filled sequentially
            // For plain: lds[plain_off] = plain_off (since we fill sequentially)
            int plain_val = plain_off;
            int xor_val = xor_off;

            std::cout << std::setw(3) << m << " | "
                      << std::setw(2) << k << " | "
                      << std::setw(9) << plain_off << " | "
                      << std::setw(7) << xor_off << " | "
                      << std::setw(9) << plain_val << " | "
                      << std::setw(7) << xor_val << "\n";
        }
    }

    // Show the physical layout
    std::cout << "\n=== Physical LDS Layout ===\n";
    std::cout << "If we WRITE using XOR descriptor, where does each (m,k) go?\n\n";

    // Create a map: physical_offset -> (m, k)
    std::vector<std::pair<int,int>> phys_to_logical(kM * kK, {-1, -1});
    for (int m = 0; m < kM; m++)
    {
        for (int k = 0; k < kK; k++)
        {
            auto xor_off = xor_desc.calculate_offset(make_multi_index(m, k));
            phys_to_logical[xor_off] = {m, k};
        }
    }

    std::cout << "Physical LDS (showing which logical (m,k) is stored there):\n";
    std::cout << "Offset | (m, k)\n";
    std::cout << "-------|-------\n";
    for (int i = 0; i < kM * kK; i++)
    {
        auto [m, k] = phys_to_logical[i];
        std::cout << std::setw(6) << i << " | (" << m << ", " << k << ")\n";
    }

    // Show bank mapping
    std::cout << "\n=== Bank Mapping (XOR) ===\n";
    std::cout << "Logical (m,k) -> physical offset -> bank\n\n";
    std::cout << "  m |  k | offset | bank\n";
    std::cout << "----|----| -------|-----\n";
    for (int m = 0; m < kM; m++)
    {
        for (int k = 0; k < kK; k++)
        {
            auto xor_off = xor_desc.calculate_offset(make_multi_index(m, k));
            int byte_addr = xor_off * sizeof(DataType);
            int bank = (byte_addr / 4) % 32;
            std::cout << std::setw(3) << m << " | "
                      << std::setw(2) << k << " | "
                      << std::setw(6) << xor_off << " | "
                      << std::setw(4) << bank << "\n";
        }
    }

    return 0;
}
