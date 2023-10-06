#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

#include "reference_gemm.hpp"
#include "gemm_gemm.hpp"

int main(int argc, char* argv[])
{
    using A0DataType   = ck::half_t;
    using B0DataType   = ck::half_t;
    using B1DataType   = ck::half_t;
    using Acc0DataType = float;
    using C0DataType   = ck::half_t;
    using Acc1DataType = float;
    using C1DataType   = ck::half_t;

    ck::index_t M0 = 13312;
    ck::index_t N0 = 4096;
    ck::index_t K0 = 128;
    ck::index_t N1 = 128;

    if(argc == 5)
    {
        M0 = std::stoi(argv[1]);
        N0 = std::stoi(argv[2]);
        K0 = std::stoi(argv[3]);
        N1 = std::stoi(argv[4]);
    }

    std::array<ck::index_t, 2> a0_lengths{M0, K0};
    std::array<ck::index_t, 2> a0_strides{K0, 1};

    std::array<ck::index_t, 2> b0_lengths{N0, K0};
    std::array<ck::index_t, 2> b0_strides{K0, 1};

    std::array<ck::index_t, 2> c0_lengths{M0, N0};
    std::array<ck::index_t, 2> c0_strides{N0, 1};

    std::array<ck::index_t, 2> b1_lengths{N1, N0};
    std::array<ck::index_t, 2> b1_strides{N0, 1};

    std::array<ck::index_t, 2> c1_lengths{M0, N1};
    std::array<ck::index_t, 2> c1_strides{N1, 1};

    // host verify
    Tensor<A0DataType> a0_host(a0_lengths, a0_strides);
    Tensor<B0DataType> b0_host(b0_lengths, b0_strides);
    Tensor<B1DataType> b1_host(b1_lengths, b1_strides);
    Tensor<C0DataType> c0_host_ref(c0_lengths, c0_strides);
    Tensor<C1DataType> c1_host_ref(c1_lengths, c1_strides);
    Tensor<C1DataType> c1_host_dev(c1_lengths, c1_strides);

    ck::utils::FillUniformDistributionIntegerValue<A0DataType>{-3.f, 3.f}(a0_host);
    ck::utils::FillUniformDistributionIntegerValue<B0DataType>{-3.f, 3.f}(b0_host);
    ck::utils::FillUniformDistributionIntegerValue<B1DataType>{-3.f, 3.f}(b1_host);

    // reference gemm
    reference_gemm<A0DataType, B0DataType, Acc0DataType, C0DataType>(a0_host, b0_host, c0_host_ref);
    reference_gemm<C0DataType, B1DataType, Acc1DataType, C1DataType>(
        c0_host_ref, b1_host, c1_host_ref);

    DeviceMem a0_buf(sizeof(A0DataType) * a0_host.GetElementSpaceSize());
    DeviceMem b0_buf(sizeof(B0DataType) * b0_host.GetElementSpaceSize());
    DeviceMem b1_buf(sizeof(B1DataType) * b1_host.GetElementSpaceSize());
    DeviceMem c1_buf(sizeof(C1DataType) * c1_host_ref.GetElementSpaceSize());

    a0_buf.ToDevice(a0_host.mData.data());
    b0_buf.ToDevice(b0_host.mData.data());
    b1_buf.ToDevice(b1_host.mData.data());

    constexpr ck::index_t kM0PerBlock = 128;
    constexpr ck::index_t kN0PerBlock = 128;
    constexpr ck::index_t kK0PerBlock = 32;
    constexpr ck::index_t kN1PerBlock = 128;
    constexpr ck::index_t kK1PerBlock = 32;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M0 / kM0PerBlock) * (N1 / kN1PerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    float ave_time =
        launch_kernel<kBlockSize, kBlockPerCu>(StreamConfig{nullptr, true},
                                               GemmGemm<A0DataType,
                                                        B0DataType,
                                                        B1DataType,
                                                        Acc0DataType,
                                                        C0DataType,
                                                        Acc1DataType,
                                                        C1DataType,
                                                        kBlockSize,
                                                        kM0PerBlock,
                                                        kN0PerBlock,
                                                        kK0PerBlock,
                                                        kN1PerBlock,
                                                        kK1PerBlock>{},
                                               kGridSize,
                                               kBlockSize,
                                               0,
                                               static_cast<A0DataType*>(a0_buf.GetDeviceBuffer()),
                                               static_cast<B0DataType*>(b0_buf.GetDeviceBuffer()),
                                               static_cast<B1DataType*>(b1_buf.GetDeviceBuffer()),
                                               static_cast<C1DataType*>(c1_buf.GetDeviceBuffer()),
                                               M0,
                                               N0,
                                               K0,
                                               N1,
                                               K0,  // Lda0
                                               K0,  // Ldb0
                                               N0,  // Ldb1
                                               N1); // Ldc1

    c1_buf.FromDevice(c1_host_dev.mData.data());

    std::size_t flop      = std::size_t(2) * M0 * N0 * K0 + std::size_t(2) * M0 * N1 * N0;
    std::size_t num_btype = sizeof(A0DataType) * M0 * K0 + sizeof(B0DataType) * N0 * K0 +
                            sizeof(B1DataType) * N1 * N0 + sizeof(C1DataType) * M0 * N1;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return !ck::utils::check_err(c1_host_dev, c1_host_ref);
}
