#pragma once

#include <type_traits>
#include <concepts>
#include <array>
#include "types.hpp"

namespace ck_tile::builder {

// Concept for thread block dimensions for a GEMM problem.
template <typename T>
concept ThreadBlockDescriptor = requires(T t) {
    { t.block_size } -> std::convertible_to<int>;
    { t.submatrix.m } -> std::convertible_to<int>;
    { t.submatrix.n } -> std::convertible_to<int>;
    { t.submatrix.k } -> std::convertible_to<int>;
};

// Concept to check if struct specifies thread block info.
template <typename T>
concept SpecifiesThreadBlock = requires {
    { T::thread_block } -> ThreadBlockDescriptor;
};

// Concept for tuning parameters for a convolution problem.
template <typename T>
concept ConvTuningDescriptor = requires(T t) {
    { t.ak1 } -> std::convertible_to<int>;
    { t.bk1 } -> std::convertible_to<int>;
    { t.m_per_xdl } -> std::convertible_to<int>;
    { t.n_per_xdl } -> std::convertible_to<int>;
    { t.m_xdl_per_wave } -> std::convertible_to<int>;
    { t.n_xdl_per_wave } -> std::convertible_to<int>;
};

// Concept to check if a struct specifies convolution tuning info.
template <typename T>
concept SpecifiesConvTuning = requires {
    { T::tuning_params } -> ConvTuningDescriptor;
};

// Concept for A block transfer thread cluster lengths.
template <typename T>
concept BlockATransferDescriptor = requires(T t) {
    { t.k0 } -> std::convertible_to<int>;
    { t.m } -> std::convertible_to<int>;
    { t.k1 } -> std::convertible_to<int>;
};

// Concept for B block transfer thread cluster lengths.
template <typename T>
concept BlockBTransferDescriptor = requires(T t) {
    { t.k0 } -> std::convertible_to<int>;
    { t.n } -> std::convertible_to<int>;
    { t.k1 } -> std::convertible_to<int>;
};

// Concept for the thread cluster access order
template <typename T>
concept ThreadClusterAccessOrderDescriptor = requires(T t) {
    { t.order } -> std::convertible_to<std::array<int, 3>>;
};

// Concept to describe source access order
template <typename T>
concept SourceAccessOrderDescriptor = requires(T t) {
    { t.order } -> std::convertible_to<std::array<int, 3>>;
}; 

// Concept for C block transfer thread cluster lengths.
template <typename T>
concept BlockCTransferDescriptor = requires(T t) {
    { t.m_block } -> std::convertible_to<int>;
    { t.m_wave_per_xdl } -> std::convertible_to<int>;
    { t.n_block } -> std::convertible_to<int>;
    { t.n_wave_per_xdl } -> std::convertible_to<int>;
};

// Concept for vector transfer details for A and B tensors
template <typename T>
concept VectorTransferDescriptorAB = requires(T t) {
    { t.src_vector_dim } -> std::convertible_to<size_t>;
    { t.src_scaler_per_vector } -> std::convertible_to<size_t>;
    { t.dest_scaler_per_vector_k1 } -> std::convertible_to<size_t>;
    { t.add_extra } -> std::convertible_to<bool>;
};

// Concept for the C tensor vectors transfer details.
template <typename T>
concept VectorTransferDescriptorC = requires(T t) {
    { t.m_xdl_per_wave_per_shuffle } -> std::convertible_to<size_t>;
    { t.n_xdl_per_wave_per_shuffle } -> std::convertible_to<size_t>;
    { t.scaler_per_vector } -> std::convertible_to<size_t>;
};

// Concept to check if a struct specifies A Block tranfer info.
template <typename T>
concept SpecifiesBlockATransfer = requires(T t) {
    { T::block_transfer.thread_cluster_dims_a } -> BlockATransferDescriptor;
};

// Concept to check if a struct specifies B Block tranfer info.
template <typename T>
concept SpecifiesBlockBTransfer = requires(T t) {
    { T::block_transfer.thread_cluster_dims_b } -> BlockBTransferDescriptor;
};

// Concept to check if a struct specifies C Block tranfer info.
template <typename T>
concept SpecifiesBlockCTransfer = requires(T t) {
    { T::block_transfer.thread_cluster_dims_c } -> BlockCTransferDescriptor;
};

// Concept to check if a struct specifies A block vector transfer info.
template <typename T>
concept SpecifiesBlockAVectorTransfer = requires(T t) {
    { T::block_transfer.vector_transfer_a } -> VectorTransferDescriptorAB;
};

// Concept to check if a struct specifies B block vector transfer info.
template <typename T>
concept SpecifiesBlockBVectorTransfer = requires(T t) {
    { T::block_transfer.vector_transfer_b } -> VectorTransferDescriptorAB;
};

// Concept to check if a struct specifies C block vector transfer info.
template <typename T>
concept SpecifiesBlockCVectorTransfer = requires(T t) {
    { T::block_transfer.vector_transfer_c } -> VectorTransferDescriptorC;
};

// Concept to check if a struct specifies thread cluster access order info.
template <typename T>
concept SpecifiesAThreadClusterAccessOrder = requires(T t) {
    { T::block_transfer.a_thread_cluster_access_order } -> ThreadClusterAccessOrderDescriptor;
};

// Concept to check if a struct specifies thread cluster access order info.
template <typename T>
concept SpecifiesBThreadClusterAccessOrder = requires(T t) {
    { T::block_transfer.b_thread_cluster_access_order } -> ThreadClusterAccessOrderDescriptor;
};

// Concept to check if a struct specifies source access order info.
template <typename T>
concept SpecifiesASourceAccessOrder = requires(T t) {
    { T::block_transfer.a_source_access_order } -> SourceAccessOrderDescriptor;
};

// Concept to check if a struct specifies source access order info.
template <typename T>
concept SpecifiesBSourceAccessOrder = requires(T t) {
    { T::block_transfer.b_source_access_order } -> SourceAccessOrderDescriptor;
};

// Concept to check if struct specifies block_gemm_pipeline_version.
template <typename T>
concept SpecifiesGemmPipelineVersion = requires {
    { T::pipeline_version } -> std::convertible_to<BlockGemmPipelineVersion>;
};

// No requirements yet for a ConvAlogorithm concept.
template <typename T>
concept ConvAlgorithmDescriptor = std::is_class_v<T>;

} // namespace ck_tile::builder
