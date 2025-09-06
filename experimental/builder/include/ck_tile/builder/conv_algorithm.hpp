#pragma once

#include <type_traits>
#include <concepts>
#include <array>

namespace ck_tile::builder {

// Convenience struct for a tuple of m, n, and k values.
template <typename T>
struct MNK
{
    T m{};
    T n{};
    T k{};
};

// Concept for thread block info for a GEMM problem.
template <typename T>
concept ThreadBlockInfo = requires(T t) {
    { t.block_size } -> std::convertible_to<int>;
    { t.submatrix.m } -> std::convertible_to<int>;
    { t.submatrix.n } -> std::convertible_to<int>;
    { t.submatrix.k } -> std::convertible_to<int>;
};

// Describe a thread block for a GEMM.
struct ThreadBlock
{
    // Thread block size.
    int block_size;
    // Size of the submatrix problem in a thread block.
    MNK<int> submatrix;
};
static_assert(ThreadBlockInfo<ThreadBlock>);

// Concept to check if struct provides thread block info.
template <typename T>
concept HasThreadBlockInfo = requires {
    { T::thread_block } -> ThreadBlockInfo;
};

// Concept for tuning parameters for a convolution problem.
template <typename T>
concept ConvTuningInfo = requires(T t) {
    { t.ak1 } -> std::convertible_to<int>;
    { t.bk1 } -> std::convertible_to<int>;
    { t.m_xdl_per_wave } -> std::convertible_to<int>;
    { t.n_xdl_per_wave } -> std::convertible_to<int>;
};

// Describe some convolution tuning parameters.
struct ConvTuningParams
{
    // NOTE: ak1 and bk1 are difficult to verify in the kernel instantiation!!!
    int ak1            = 0;
    int bk1            = 0;
    int m_xdl_per_wave = 0;
    int n_xdl_per_wave = 0;
};
static_assert(ConvTuningInfo<ConvTuningParams>);

// Concept to check if a struct provides convolution tuning info.
template <typename T>
concept HasConvTuningInfo = requires {
    { T::tuning_params } -> ConvTuningInfo;
};

// Concept for A block transfer thread cluster lengths.
template <typename T>
concept BlockATransferLengths = requires(T t) {
    { t.k0 } -> std::convertible_to<int>;
    { t.m } -> std::convertible_to<int>;
    { t.k1 } -> std::convertible_to<int>;
};

// Describe A block transfer thread cluster lengths.
struct BlockATransferLengthsInfo
{
    int k0;
    int m;
    int k1;
};
static_assert(BlockATransferLengths<BlockATransferLengthsInfo>);

// Concept for B block transfer thread cluster lengths.
template <typename T>
concept BlockBTransferLengths = requires(T t) {
    { t.k0 } -> std::convertible_to<int>;
    { t.n } -> std::convertible_to<int>;
    { t.k1 } -> std::convertible_to<int>;
};

// Describe B block transfer thread cluster lengths.
struct BlockBTransferLengthsInfo
{
    int k0;
    int n;
    int k1;
};
static_assert(BlockBTransferLengths<BlockBTransferLengthsInfo>);

// Concept for C block transfer thread cluster lengths.
template <typename T>
concept BlockCTransferLengths = requires(T t) {
    { t.m_block } -> std::convertible_to<int>;
    { t.m_wave_per_xdl } -> std::convertible_to<int>;
    { t.n_block } -> std::convertible_to<int>;
    { t.n_wave_per_xdl } -> std::convertible_to<int>;
};

// Describe C block transfer thread cluster lengths.
struct BlockCTransferLengthsInfo
{
    int m_block;
    int m_wave_per_xdl;
    int n_block;
    int n_wave_per_xdl;
};
static_assert(BlockCTransferLengths<BlockCTransferLengthsInfo>);

// Concept to check if a struct provides A Block tranfer info.
template <typename T>
concept HasABlockTransferInfo = requires(T t) {
    { T::block_transfer.thread_cluster_dims_a } -> BlockATransferLengths;
};

// Concept to check if a struct provides B Block tranfer info.
template <typename T>
concept HasBBlockTransferInfo = requires(T t) {
    { T::block_transfer.thread_cluster_dims_b } -> BlockBTransferLengths;
};

// Concept to check if a struct provides C Block tranfer info.
template <typename T>
concept HasCBlockTransferInfo = requires(T t) {
    { T::block_transfer.thread_cluster_dims_c } -> BlockCTransferLengths;
};

enum class BlockGemmPipelineVersion
{
    V1,
    V3,
    V4,
    V5
};

// Concept to check if struct provides block_gemm_pipeline_version.
template <typename T>
concept ProvidesBlockGemmPipelineVersion = requires {
    { T::pipeline_version } -> std::convertible_to<BlockGemmPipelineVersion>;
};

// No requirements yet for a ConvAlogorithm concept.
template <typename T>
concept ConvAlgorithm = std::is_class_v<T>;

} // namespace ck_tile::builder
