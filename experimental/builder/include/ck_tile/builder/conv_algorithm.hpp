#pragma once

#include <type_traits>
#include <concepts>

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
    { t.sub_matrix.m } -> std::convertible_to<int>;
    { t.sub_matrix.n } -> std::convertible_to<int>;
    { t.sub_matrix.k } -> std::convertible_to<int>;
};

// Describe a thread block for a GEMM.
struct ThreadBlock
{
    // Thread block size.
    int block_size;
    // Size of the submatrix problem in a thread block.
    MNK<int> sub_matrix;
};

static_assert(ThreadBlockInfo<ThreadBlock>);

// Concept to check if struct provides thread block info.
template <typename T>
concept HasThreadBlockInfo = requires {
    { T::THREAD_BLOCK } -> ThreadBlockInfo;
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
    { T::TUNING_PARAMS } -> ConvTuningInfo;
};

// No requirements yet for a ConvAlogorithm concept.
template <typename T>
concept ConvAlgorithm = std::is_class_v<T>;

} // namespace ck_tile::builder
