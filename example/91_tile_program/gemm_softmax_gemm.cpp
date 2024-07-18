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
#include "reference_softmax.hpp"
#include "gemm_softmax_gemm.hpp"

int main(int argc, char* argv[])
{
    using QDataType           = ck::half_t;
    using KDataType           = ck::half_t;
    using VDataType           = ck::half_t;
    using SaccDataType        = float;
    using SMPLComputeDataType = float;
    using PDataType           = ck::half_t;
    using OaccDataType        = float;
    using ODataType           = ck::half_t;

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

    std::array<ck::index_t, 2> q_lengths{M0, K0};
    std::array<ck::index_t, 2> q_strides{K0, 1};

    std::array<ck::index_t, 2> k_lengths{N0, K0};
    std::array<ck::index_t, 2> k_strides{K0, 1};

    std::array<ck::index_t, 2> v_lengths{N1, N0};
    std::array<ck::index_t, 2> v_strides{N0, 1};

    std::array<ck::index_t, 2> s_lengths{M0, N0};
    std::array<ck::index_t, 2> s_strides{N0, 1};

    std::array<ck::index_t, 2> p_lengths{M0, N0};
    std::array<ck::index_t, 2> p_strides{N0, 1};

    std::array<ck::index_t, 2> o_lengths{M0, N1};
    std::array<ck::index_t, 2> o_strides{N1, 1};

    // host verify
    Tensor<QDataType> q_host(q_lengths, q_strides);
    Tensor<KDataType> k_host(k_lengths, k_strides);
    Tensor<VDataType> v_host(v_lengths, v_strides);
    Tensor<SMPLComputeDataType> s_host_ref(s_lengths, s_strides);
    Tensor<PDataType> p_host_ref(p_lengths, p_strides);
    Tensor<ODataType> o_host_ref(o_lengths, o_strides);
    Tensor<ODataType> o_host_dev(o_lengths, o_strides);

#if 0
    ck::utils::FillUniformDistributionIntegerValue<QDataType>{-3.f, 3.f}(q_host);
    ck::utils::FillUniformDistributionIntegerValue<KDataType>{-3.f, 3.f}(k_host);
    ck::utils::FillUniformDistributionIntegerValue<VDataType>{-3.f, 3.f}(v_host);
#else
    ck::utils::FillUniformDistribution<QDataType>{-3.f, 3.f}(q_host);
    ck::utils::FillUniformDistribution<KDataType>{-3.f, 3.f}(k_host);
    ck::utils::FillUniformDistribution<VDataType>{-3.f, 3.f}(v_host);
#endif

    // reference
    reference_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
        q_host, k_host, s_host_ref);
    reference_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref, p_host_ref);
    reference_gemm<PDataType, VDataType, OaccDataType, ODataType>(p_host_ref, v_host, o_host_ref);

    DeviceMem q_buf(sizeof(QDataType) * q_host.GetElementSpaceSize());
    DeviceMem k_buf(sizeof(KDataType) * k_host.GetElementSpaceSize());
    DeviceMem v_buf(sizeof(VDataType) * v_host.GetElementSpaceSize());
    DeviceMem o_buf(sizeof(ODataType) * o_host_ref.GetElementSpaceSize());

    q_buf.ToDevice(q_host.mData.data());
    k_buf.ToDevice(k_host.mData.data());
    v_buf.ToDevice(v_host.mData.data());

    constexpr ck::index_t kM0PerBlock = 128;
    constexpr ck::index_t kN0PerBlock = 128;
    constexpr ck::index_t kK0PerBlock = 32;
    constexpr ck::index_t kN1PerBlock = 128;

    constexpr ck::index_t kBlockSize = 256;
    ck::index_t kGridSize            = (M0 / kM0PerBlock) * (N1 / kN1PerBlock);

    std::cout << "grid size " << kGridSize << std::endl;

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    float ave_time =
        launch_kernel<kBlockSize, kBlockPerCu>(StreamConfig{nullptr, true},
                                               GemmSoftmaxGemm<QDataType,
                                                               KDataType,
                                                               VDataType,
                                                               SaccDataType,
                                                               SMPLComputeDataType,
                                                               PDataType,
                                                               OaccDataType,
                                                               ODataType,
                                                               kBlockSize,
                                                               kM0PerBlock,
                                                               kN0PerBlock,
                                                               kK0PerBlock,
                                                               kN1PerBlock>{},
                                               kGridSize,
                                               kBlockSize,
                                               0,
                                               static_cast<QDataType*>(q_buf.GetDeviceBuffer()),
                                               static_cast<KDataType*>(k_buf.GetDeviceBuffer()),
                                               static_cast<VDataType*>(v_buf.GetDeviceBuffer()),
                                               static_cast<ODataType*>(o_buf.GetDeviceBuffer()),
                                               M0,
                                               N0,
                                               K0,
                                               N1,
                                               K0,  // StrideQ
                                               K0,  // StrideK
                                               N0,  // StrideV
                                               N1); // StrideO

    o_buf.FromDevice(o_host_dev.mData.data());

    std::size_t flop      = std::size_t(2) * M0 * N0 * K0 + std::size_t(2) * M0 * N1 * N0;
    std::size_t num_btype = sizeof(QDataType) * M0 * K0 + sizeof(KDataType) * N0 * K0 +
                            sizeof(VDataType) * N1 * N0 + sizeof(ODataType) * M0 * N1;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return !ck::utils::check_err(o_host_dev, o_host_ref);
}
