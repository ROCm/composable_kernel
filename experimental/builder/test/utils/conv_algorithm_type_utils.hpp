// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../impl/conv_algorithm_types.hpp"
#include <sstream>
#include <array>

namespace ck_tile::builder::test {

namespace ckb = ck_tile::builder;

// Helper function to convert arrays to Seq(...) format
template <typename T, size_t N>
std::string array_to_seq(const std::array<T, N>& arr)
{
    std::ostringstream oss;
    oss << "Seq(";
    for(size_t i = 0; i < N; ++i)
    {
        if(i > 0)
            oss << ",";
        oss << arr[i];
    }
    oss << ")";
    return oss.str();
}

// Base template - will cause compilation error for unsupported types
template <typename T>
std::string to_string(T)
{
    static_assert(sizeof(T) == 0, "Unsupported type");
    return "";
}

// Template specializations for enum types

template <>
inline std::string to_string<PipelineVersion>(PipelineVersion t)
{
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

template <>
inline std::string to_string<PipelineScheduler>(PipelineScheduler t)
{
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

template <>
inline std::string to_string<ConvSpecialization>(ConvSpecialization t)
{
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

template <>
inline std::string to_string<GemmSpecialization>(GemmSpecialization t)
{
    std::ostringstream oss;
    oss << t;
    return oss.str();
}

// Template specializations for struct types

template <>
inline std::string to_string<MNK<size_t>>(MNK<size_t> t)
{
    return array_to_seq(std::array<size_t, 3>{t.m, t.n, t.k});
}

template <>
inline std::string to_string<ThreadBlock>(ThreadBlock t)
{
    std::ostringstream oss;
    oss << t.block_size << "," << t.tile_size.m << "," << t.tile_size.n << "," << t.tile_size.k;
    return oss.str();
}

template <>
inline std::string to_string<GridwiseBwdDataXdlGemm>(GridwiseBwdDataXdlGemm t)
{
    std::ostringstream oss;
    oss << t.ak1 << "," << t.bk1 << "," << t.xdl_params.m_per_xdl << "," << t.xdl_params.n_per_xdl
        << "," << t.xdl_params.m_xdl_per_wave << "," << t.xdl_params.n_xdl_per_wave;
    return oss.str();
}

template <>
inline std::string to_string<GridwiseBwdXdlGemm>(GridwiseBwdXdlGemm t)
{
    std::ostringstream oss;
    oss << t.k1 << "," << t.xdl_params.m_per_xdl << "," << t.xdl_params.n_per_xdl << ","
        << t.xdl_params.m_xdl_per_wave << "," << t.xdl_params.n_xdl_per_wave;
    return oss.str();
}

template <>
inline std::string to_string<GridwiseFwdXdlGemm>(GridwiseFwdXdlGemm t)
{
    std::ostringstream oss;
    oss << t.ak1 << "," << t.bk1 << "," << t.xdl_params.m_per_xdl << "," << t.xdl_params.n_per_xdl
        << "," << t.xdl_params.m_xdl_per_wave << "," << t.xdl_params.n_xdl_per_wave;
    return oss.str();
}

template <>
inline std::string to_string<GridwiseWmmaGemm>(GridwiseWmmaGemm t)
{
    std::ostringstream oss;
    oss << t.k1 << "," << t.m_per_wmma << "," << t.n_per_wmma << "," << t.m_wmma_per_wave << ","
        << t.n_wmma_per_wave;
    return oss.str();
}

template <>
inline std::string to_string<GridwiseWmmaGemmABK1>(GridwiseWmmaGemmABK1 t)
{
    std::ostringstream oss;
    oss << t.ak1 << "," << t.bk1 << "," << t.m_per_wmma << "," << t.n_per_wmma << ","
        << t.m_wmma_per_wave << "," << t.n_wmma_per_wave;
    return oss.str();
}

template <>
inline std::string to_string<BlockGemmPipeline>(BlockGemmPipeline t)
{
    std::ostringstream oss;
    oss << to_string(t.scheduler) << "," << to_string(t.pipeline_version);
    return oss.str();
}

template <size_t ThreadClusterRank>
inline std::string to_string(BlockTransfer<ThreadClusterRank> t)
{
    if constexpr(ThreadClusterRank == 4)
    {
        return array_to_seq(std::array<size_t, 4>{t.k_batch_size, t.k0, t.m_n, t.k1});
    }
    else if constexpr(ThreadClusterRank == 3)
    {
        return array_to_seq(std::array<size_t, 3>{t.k0, t.m_n, t.k1});
    }
    else
    {
        static_assert(ThreadClusterRank == 3 || ThreadClusterRank == 4,
                      "Unsupported ThreadClusterRank");
    }
}

template <>
inline std::string to_string<ThreadCluster>(ThreadCluster t)
{
    return array_to_seq(
        std::array<size_t, 4>{t.m_block, t.m_wave_per_xdl, t.n_block, t.n_wave_per_xdl});
}

template <>
inline std::string to_string<LdsTransfer>(LdsTransfer t)
{
    std::ostringstream oss;
    oss << t.src_vector_dim << "," << t.src_scalar_per_vector << "," << t.lds_dst_scalar_per_vector
        << "," << (t.lds_padding ? "true" : "false") << ","
        << (t.is_direct_load ? "true" : "false");
    return oss.str();
}

template <size_t N>
inline std::string to_string(AccessOrder<N> t)
{
    return array_to_seq(t.order);
}

template <size_t N = 3>
inline std::string to_string(InputTransfer<N> t)
{
    std::ostringstream oss;
    oss << to_string(t.block_transfer) << "," << to_string(t.thread_cluster_arrange_order) << ","
        << to_string(t.src_access_order) << "," << t.lds_transfer.src_vector_dim << ","
        << t.lds_transfer.src_scalar_per_vector << "," << t.lds_transfer.lds_dst_scalar_per_vector
        << "," << (t.lds_transfer.lds_padding ? "true" : "false");
    return oss.str();
}

template <>
inline std::string to_string<OutputTransfer>(OutputTransfer t)
{
    std::ostringstream oss;
    oss << t.epilogue.m_xdl_per_wave_per_shuffle << "," << t.epilogue.n_xdl_per_wave_per_shuffle
        << "," << to_string(t.thread_cluster_dims) << "," << t.epilogue.scalar_per_vector;
    return oss.str();
}

template <size_t N = 3>
inline std::string to_string(Transfer<N> t)
{
    std::ostringstream oss;
    oss << to_string(t.a) << "," << to_string(t.b) << "," << to_string(t.c);
    return oss.str();
}

template <>
inline std::string to_string<DlThreadConfig>(DlThreadConfig t)
{
    std::ostringstream oss;
    oss << t.k1 << "," << t.m1_per_thread << "," << t.n1_per_thread << "," << t.k_per_thread;
    return oss.str();
}

template <>
inline std::string to_string<DlThreadCluster>(DlThreadCluster t)
{
    std::ostringstream oss;
    oss << array_to_seq(t.m1_xs) << "," << array_to_seq(t.n1_xs);
    return oss.str();
}

template <>
inline std::string to_string<DlBlockTransfer<4>>(DlBlockTransfer<4> t)
{
    std::ostringstream oss;
    oss << array_to_seq(t.thread_slice_lengths) << "," << array_to_seq(t.thread_cluster_lengths)
        << "," << array_to_seq(t.thread_cluster_arrange_order) << ","
        << array_to_seq(t.src_access_order) << "," << array_to_seq(t.src_vector_tensor_lengths)
        << "," << array_to_seq(t.src_vector_tensor_contiguous_dim_order) << ","
        << array_to_seq(t.dst_vector_tensor_lengths);
    return oss.str();
}

template <>
inline std::string to_string<DlBlockTransfer<5>>(DlBlockTransfer<5> t)
{
    std::ostringstream oss;
    oss << array_to_seq(t.thread_slice_lengths) << "," << array_to_seq(t.thread_cluster_lengths)
        << "," << array_to_seq(t.thread_cluster_arrange_order) << ","
        << array_to_seq(t.src_access_order) << "," << array_to_seq(t.src_vector_tensor_lengths)
        << "," << array_to_seq(t.src_vector_tensor_contiguous_dim_order) << ","
        << array_to_seq(t.dst_vector_tensor_lengths);
    return oss.str();
}

template <>
inline std::string to_string<DlEpilogue>(DlEpilogue t)
{
    std::ostringstream oss;
    oss << array_to_seq(t.src_dst_access_order) << "," << t.src_dst_vector_dim << ","
        << t.dst_scalar_per_vector;
    return oss.str();
}

template <>
inline std::string to_string<TransposeParams_>(TransposeParams_ t)
{
    std::ostringstream oss;
    oss << t.max_transpose_transfer_src_scalar_per_vector << ","
        << t.max_transpose_transfer_dst_scalar_per_vector;
    return oss.str();
}

template <>
inline std::string to_string<DlTransfer<4>>(DlTransfer<4> t)
{
    std::ostringstream oss;
    oss << to_string(t.a) << "," << to_string(t.b) << "," << to_string(t.c);
    return oss.str();
}

template <>
inline std::string to_string<DlTransfer<5>>(DlTransfer<5> t)
{
    std::ostringstream oss;
    oss << to_string(t.a) << "," << to_string(t.b) << "," << to_string(t.c);
    return oss.str();
}

// Template specializations for factory wrapper types

template <>
inline std::string to_string<ThreadBlock_>(ThreadBlock_ t)
{
    return to_string(t.thread_block);
}

template <>
inline std::string to_string<FwdXdlGemm_>(FwdXdlGemm_ t)
{
    return to_string(t.gridwise_gemm);
}

template <>
inline std::string to_string<BwdXdlGemm_>(BwdXdlGemm_ t)
{
    return to_string(t.gridwise_gemm);
}

template <>
inline std::string to_string<BwdDataXdlGemm_>(BwdDataXdlGemm_ t)
{
    return to_string(t.gridwise_gemm);
}

template <>
inline std::string to_string<WmmaGemm_>(WmmaGemm_ t)
{
    return to_string(t.gridwise_gemm);
}

template <>
inline std::string to_string<WmmaGemmABK1_>(WmmaGemmABK1_ t)
{
    return to_string(t.gridwise_gemm);
}

template <size_t ThreadClusterRank = 3>
inline std::string to_string(Transfer_<ThreadClusterRank> t)
{
    return to_string(t.transfer);
}

template <>
inline std::string to_string<ConvSpecializationFwd_>(ConvSpecializationFwd_ t)
{
    std::ostringstream oss;
    oss << to_string(t.fwd_specialization) << "," << to_string(t.gemm_specialization);
    return oss.str();
}

template <>
inline std::string to_string<ConvSpecializationBwdWeight_>(ConvSpecializationBwdWeight_ t)
{
    std::ostringstream oss;
    oss << to_string(t.bwd_weight_specialization);
    return oss.str();
}

template <>
inline std::string to_string<ConvSpecializationBwdData_>(ConvSpecializationBwdData_ t)
{
    std::ostringstream oss;
    oss << to_string(t.bwd_data_specialization);
    return oss.str();
}

template <>
inline std::string to_string<Prefetch_>(Prefetch_ t)
{
    std::ostringstream oss;
    oss << t.num_gemm_k_prefetch_stages << "," << to_string(t.loop_scheduler);
    return oss.str();
}

template <>
inline std::string to_string<BlockGemm_>(BlockGemm_ t)
{
    return to_string(t.block_gemm_pipeline);
}

template <>
inline std::string to_string<DlThreadConfig_>(DlThreadConfig_ t)
{
    return to_string(t.thread_config);
}

template <>
inline std::string to_string<DlThreadCluster_>(DlThreadCluster_ t)
{
    return to_string(t.thread_cluster);
}

template <>
inline std::string to_string<DlTransfer_<4>>(DlTransfer_<4> t)
{
    return to_string(t.transfer);
}

template <>
inline std::string to_string<DlTransfer_<5>>(DlTransfer_<5> t)
{
    return to_string(t.transfer);
}

// Template specializations for algorithm types

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<FwdXdlGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<FwdXdlGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << to_string(static_cast<WmmaGemmABK1_>(t)) << ","
        << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<WmmaGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK>(
    ConvAlgorithm_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << to_string(static_cast<DlThreadConfig_>(t)) << ","
        << to_string(static_cast<DlThreadCluster_>(t)) << ","
        << to_string(static_cast<DlTransfer_<4>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor>(
    ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<FwdXdlGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<BwdXdlGemm_>(t))
        << "," << to_string(static_cast<Transfer_<4>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<BwdXdlGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<WmmaGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<WmmaGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<WmmaGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_Wmma_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_Wmma_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<WmmaGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_Xdl_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_Xdl_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<BwdXdlGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_Dl>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_Dl t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << to_string(static_cast<DlThreadConfig_>(t)) << ","
        << to_string(static_cast<DlThreadCluster_>(t)) << ","
        << to_string(static_cast<DlTransfer_<5>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Xdl_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<BwdXdlGemm_>(t))
        << "," << to_string(static_cast<Transfer_<4>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << to_string(static_cast<BwdDataXdlGemm_>(t)) << ","
        << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << "," << to_string(static_cast<WmmaGemm_>(t))
        << "," << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << to_string(static_cast<WmmaGemmABK1_>(t)) << ","
        << to_string(static_cast<Transfer_<>>(t));
    return oss.str();
}

} // namespace ck_tile::builder::test
