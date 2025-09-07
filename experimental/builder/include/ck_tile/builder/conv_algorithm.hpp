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

// Concept for thread block dimensions for a GEMM problem.
template <typename T>
concept ThreadBlockDescriptor = requires(T t) {
    { t.block_size } -> std::convertible_to<int>;
    { t.submatrix.m } -> std::convertible_to<int>;
    { t.submatrix.n } -> std::convertible_to<int>;
    { t.submatrix.k } -> std::convertible_to<int>;
};

// Specifiy thread block dimensions for a GEMM.
struct ThreadBlock
{
    // Thread block size.
    int block_size;
    // Size of the submatrix problem in a thread block.
    MNK<int> submatrix;
};
static_assert(ThreadBlockDescriptor<ThreadBlock>);

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
static_assert(ConvTuningDescriptor<ConvTuningParams>);

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

// Describe A block transfer thread cluster lengths.
struct BlockATransferLengths
{
    int k0;
    int m;
    int k1;
};
static_assert(BlockATransferDescriptor<BlockATransferLengths>);

// Concept for B block transfer thread cluster lengths.
template <typename T>
concept BlockBTransferDescriptor = requires(T t) {
    { t.k0 } -> std::convertible_to<int>;
    { t.n } -> std::convertible_to<int>;
    { t.k1 } -> std::convertible_to<int>;
};

// Describe B block transfer thread cluster lengths.
struct BlockBTransferLengths
{
    int k0;
    int n;
    int k1;
};
static_assert(BlockBTransferDescriptor<BlockBTransferLengths>);

// Concept for C block transfer thread cluster lengths.
template <typename T>
concept BlockCTransferDescriptor = requires(T t) {
    { t.m_block } -> std::convertible_to<int>;
    { t.m_wave_per_xdl } -> std::convertible_to<int>;
    { t.n_block } -> std::convertible_to<int>;
    { t.n_wave_per_xdl } -> std::convertible_to<int>;
};

// Describe C block transfer thread cluster lengths.
struct BlockCTransferLengths
{
    int m_block;
    int m_wave_per_xdl;
    int n_block;
    int n_wave_per_xdl;
};
static_assert(BlockCTransferDescriptor<BlockCTransferLengths>);

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

// Enums for the current block GEMM pipeline versions.
enum class BlockGemmPipelineVersion
{
    V1,
    V3,
    V4,
    V5
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
