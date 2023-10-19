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

#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qkvs_default_policy.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp"
#include "ck/tile_program/tile/tile_fmha_shape.hpp"

#include "reference_batched_gemm.hpp"
#include "reference_batched_softmax.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_epilogue.hpp"

using QDataType           = ck::half_t;
using KDataType           = ck::half_t;
using VDataType           = ck::half_t;
using SaccDataType        = float;      // data type for first gemm accumulation
using SMPLComputeDataType = float;      // data type for reduction, softmax
using PDataType           = ck::half_t; // data type for A matrix of second gemm
using OaccDataType        = float;      // data type for second gemm accumulation
using ODataType           = ck::half_t;

using FmhaShape =
    ck::tile_program::TileFmhaShape<128 /*M0*/, 128 /*N0*/, 32 /*K0*/, 128 /*N1*/, 32 /*K1*/>;

using FmhaPipelineProblem = ck::tile_program::block::BlockFmhaPipelineProblem<QDataType,
                                                                              KDataType,
                                                                              VDataType,
                                                                              SaccDataType,
                                                                              SMPLComputeDataType,
                                                                              PDataType,
                                                                              OaccDataType,
                                                                              ODataType,
                                                                              256, // BlockSize
                                                                              FmhaShape>;
using FmhaPipeline        = ck::tile_program::block::BlockFmhaPipelineQKVS<FmhaPipelineProblem>;

using FmhaEpilogue = FmhaFwdEpilogue<FmhaFwdEpilogueProblem<OaccDataType, ODataType>>;
using FmhaKernel   = FmhaFwdKernel<FmhaPipeline, FmhaEpilogue>;

int main(int argc, char* argv[])
{
    ck::index_t Batch = 16;   // batch * nheads
    ck::index_t M0    = 3328; // seqlen_q
    ck::index_t N0    = 4096; // seqlen_k
    ck::index_t K0    = 128;  // hdim_q
    ck::index_t N1    = 128;  // hdim_v

    if(argc == 6)
    {
        Batch = std::stoi(argv[1]);
        M0    = std::stoi(argv[2]);
        N0    = std::stoi(argv[3]);
        K0    = std::stoi(argv[4]);
        N1    = std::stoi(argv[5]);
    }

    std::array<ck::index_t, 3> q_lengths{Batch, M0, K0};
    std::array<ck::index_t, 3> q_strides{M0 * K0, K0, 1};

    std::array<ck::index_t, 3> k_lengths{Batch, N0, K0};
    std::array<ck::index_t, 3> k_strides{N0 * K0, K0, 1};

    std::array<ck::index_t, 3> v_lengths{Batch, N1, N0};
    std::array<ck::index_t, 3> v_strides{N1 * N0, N0, 1};

    std::array<ck::index_t, 3> s_lengths{Batch, M0, N0};
    std::array<ck::index_t, 3> s_strides{M0 * N0, N0, 1};

    std::array<ck::index_t, 3> p_lengths{Batch, M0, N0};
    std::array<ck::index_t, 3> p_strides{M0 * N0, N0, 1};

    std::array<ck::index_t, 3> o_lengths{Batch, M0, N1};
    std::array<ck::index_t, 3> o_strides{M0 * N1, N1, 1};

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
    reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
        q_host, k_host, s_host_ref);
    reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref,
                                                                                   p_host_ref);
    reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(
        p_host_ref, v_host, o_host_ref);

    DeviceMem q_buf(sizeof(QDataType) * q_host.GetElementSpaceSize());
    DeviceMem k_buf(sizeof(KDataType) * k_host.GetElementSpaceSize());
    DeviceMem v_buf(sizeof(VDataType) * v_host.GetElementSpaceSize());
    DeviceMem o_buf(sizeof(ODataType) * o_host_ref.GetElementSpaceSize());

    q_buf.ToDevice(q_host.mData.data());
    k_buf.ToDevice(k_host.mData.data());
    v_buf.ToDevice(v_host.mData.data());

    dim3 kGridSize            = FmhaKernel::GridSize(Batch, M0, N1);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

    std::cout << "batch:" << Batch << ", seqlen_q:" << M0 << ", seqlen_k:" << N0
              << ", hdim_q:" << K0 << ", hdim_v:" << N1 << ", grid_size " << kGridSize.x
              << std::endl;

    constexpr ck::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    constexpr ck::index_t kWarpPerBlock = kBlockSize.x / warpSize;
    constexpr ck::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;

    auto kargs = FmhaKernel::MakeKargs(q_buf.GetDeviceBuffer(),
                                       k_buf.GetDeviceBuffer(),
                                       v_buf.GetDeviceBuffer(),
                                       o_buf.GetDeviceBuffer(),
                                       M0,       // seqlen_q
                                       N0,       // seqlen_k
                                       K0,       // hdim_q
                                       N1,       // hdim_v
                                       K0,       // stride_q
                                       K0,       // stride_k
                                       N0,       // stride_v
                                       N1,       // stride_o
                                       M0 * K0,  // batch_stride_q
                                       N0 * K0,  // batch_stride_k
                                       N1 * N0,  // batch_stride_v
                                       M0 * N1); // batch_stride_o

    float ave_time = launch_kernel<kBlockSize.x, kBlockPerCu>(StreamConfig{nullptr, true},
                                                              FmhaKernel{},
                                                              kGridSize,
                                                              kBlockSize,
                                                              0,
                                                              kargs); // BatchStrideO

    o_buf.FromDevice(o_host_dev.mData.data());

    std::size_t flop =
        std::size_t(2) * Batch * M0 * N0 * K0 + std::size_t(2) * Batch * M0 * N1 * N0;

    std::size_t num_btype =
        sizeof(QDataType) * Batch * M0 * K0 + sizeof(KDataType) * Batch * N0 * K0 +
        sizeof(VDataType) * Batch * N1 * N0 + sizeof(ODataType) * Batch * M0 * N1;

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    return !ck::utils::check_err(o_host_dev, o_host_ref);
}
