#pragma once

#include <type_traits>
#include <concepts>

namespace ck_tile::builder {

// Convenience struct for a tuple of m, n, and k values.    
template <typename T>
struct MNK {
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
struct ThreadBlock {
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


// No requirements yet for a ConvAlogorithm concept.
template <typename T>
concept ConvAlgorithm = std::is_class_v<T>;

} // namespace ck_tile::builder
