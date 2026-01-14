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
inline std::string to_string<WarpGemmParams>(WarpGemmParams t)
{
    std::ostringstream oss;
    oss << t.gemm_m_per_instruction << "," << t.gemm_n_per_instruction << ","
        << t.gemm_m_iters_per_wave << "," << t.gemm_n_iters_per_wave;
    return oss.str();
}

template <>
inline std::string to_string<GemmPipeline>(GemmPipeline t)
{
    std::ostringstream oss;
    oss << t.num_gemm_k_prefetch_stages << "," << t.num_conv_groups_to_merge << ","
        << to_string(t.scheduler) << "," << to_string(t.pipeline_version);
    return oss.str();
}

template <size_t ThreadClusterRank>
inline std::string to_string(InputThreadCluster<ThreadClusterRank> t)
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
inline std::string to_string<OutputThreadCluster>(OutputThreadCluster t)
{
    return array_to_seq(std::array<size_t, 4>{
        t.gemm_m_block_size, t.gemm_m_per_block, t.gemm_n_block_size, t.gemm_n_per_block});
}

template <>
inline std::string to_string<LdsTransfer>(LdsTransfer t)
{
    std::ostringstream oss;
    oss << t.global_memory_vector_load_size << "," << t.src_vector_dim << ","
        << t.src_scalar_per_vector << "," << t.lds_dst_scalar_per_vector << ","
        << (t.lds_padding ? "true" : "false") << "," << (t.is_direct_load ? "true" : "false");
    return oss.str();
}

template <size_t N>
inline std::string to_string(AccessOrder<N> t)
{
    return array_to_seq(t.order);
}

template <size_t N = 3>
inline std::string to_string(InputTileTransfer<N> t)
{
    std::ostringstream oss;
    oss << to_string(t.thread_cluster) << "," << to_string(t.thread_cluster_access_order) << ","
        << to_string(t.src_access_order) << "," << t.lds_transfer.src_vector_dim << ","
        << t.lds_transfer.src_scalar_per_vector << "," << t.lds_transfer.lds_dst_scalar_per_vector
        << "," << (t.lds_transfer.lds_padding ? "true" : "false");
    return oss.str();
}

template <>
inline std::string to_string<OutputTileTransfer>(OutputTileTransfer t)
{
    std::ostringstream oss;
    oss << t.epilogue.m_xdl_per_wave_per_shuffle << "," << t.epilogue.n_per_wave_per_shuffle << ","
        << to_string(t.thread_cluster) << "," << t.epilogue.scalar_per_vector;
    return oss.str();
}

template <size_t N = 3>
inline std::string to_string(InputOutputTileTransfer<N> t)
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
inline std::string to_string<WarpGemm_>(WarpGemm_ t)
{
    return to_string(t.warp_gemm);
}

template <size_t ThreadClusterRank = 3>
inline std::string to_string(InputOutputTileTransfer_<ThreadClusterRank> t)
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
inline std::string to_string<GemmPipeline_>(GemmPipeline_ t)
{
    return to_string(t.gemm_pipeline);
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
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_CShuffle t)
{
    std::ostringstream oss;
    if(t.warp_gemm.matrix_instruction == MatrixInstructionType::WMMA)
    {
        oss << to_string(static_cast<ThreadBlock_>(t)) << ","
            << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
            << to_string(static_cast<WarpGemm_>(t)) << ","
            << to_string(static_cast<InputOutputTileTransfer_<>>(t));
    }
    else
    {
        oss << to_string(static_cast<ThreadBlock_>(t)) << ","
            << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
            << t.transfer.b.lds_transfer.global_memory_vector_load_size << ","
            << to_string(static_cast<WarpGemm_>(t)) << ","
            << to_string(static_cast<InputOutputTileTransfer_<>>(t));
    }
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
        << t.transfer.b.lds_transfer.global_memory_vector_load_size << ","
        << to_string(static_cast<WarpGemm_>(t)) << ","
        << to_string(static_cast<InputOutputTileTransfer_<>>(t));
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
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
        << t.transfer.b.lds_transfer.global_memory_vector_load_size << ","
        << to_string(static_cast<WarpGemm_>(t)) << ","
        << to_string(static_cast<InputOutputTileTransfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
        << to_string(static_cast<WarpGemm_>(t)) << ","
        << to_string(static_cast<InputOutputTileTransfer_<4>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
        << to_string(static_cast<WarpGemm_>(t)) << ","
        << to_string(static_cast<InputOutputTileTransfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
        << to_string(static_cast<WarpGemm_>(t)) << ","
        << to_string(static_cast<InputOutputTileTransfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
        << to_string(static_cast<WarpGemm_>(t)) << ","
        << to_string(static_cast<InputOutputTileTransfer_<>>(t));
    return oss.str();
}

template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
        << to_string(static_cast<WarpGemm_>(t)) << ","
        << to_string(static_cast<InputOutputTileTransfer_<>>(t));
    return oss.str();
}

// Covers both XDL and WMMA versions
template <>
inline std::string to_string<ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_CShuffle_V3>(
    ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_CShuffle_V3 t)
{
    std::ostringstream oss;
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
        << to_string(static_cast<WarpGemm_>(t)) << ","
        << to_string(static_cast<InputOutputTileTransfer_<>>(t));
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
    oss << to_string(static_cast<ThreadBlock_>(t)) << ","
        << t.transfer.a.lds_transfer.global_memory_vector_load_size << ","
        << to_string(static_cast<WarpGemm_>(t)) << ","
        << to_string(static_cast<InputOutputTileTransfer_<4>>(t));
    return oss.str();
}

} // namespace ck_tile::builder::test
