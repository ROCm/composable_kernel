#pragma once
#include <iostream>
#include <concepts>

namespace ck_tile::builder {

using index_t = std::size_t;

// Some sample host args to describe a GEMM, with defaults set to zero.
struct GemmHostArgs
{
    index_t m        = 0;
    index_t n        = 0;
    index_t k        = 0;
    index_t lda      = 0;
    index_t ldb      = 0;
    index_t ldc      = 0;
    index_t k_batch_ = 0;
    const void* a    = nullptr;
    const void* b    = nullptr;
    void* c          = nullptr;
};

// Tag for column major layout.
struct ColMajor
{
};

// Tag for row major layout.
struct RowMajor
{
};

// Requirements for struct to define the data types used in the GEMM operation.
//
// Example that satifies this constraint:
// struct GemmTypes {
//     using ADataType = float;
//     using BDataType = float;
//     using CDataType = float;
//     using AccDataType = float;
// };
template <typename T>
concept DefinesGemmTypes =
    requires {
        typename T::ADataType;
        typename T::BDataType;
        typename T::CDataType;
        typename T::AccDataType;
    } && std::is_arithmetic_v<typename T::ADataType> &&
    std::is_arithmetic_v<typename T::BDataType> && std::is_arithmetic_v<typename T::CDataType> &&
    std::is_arithmetic_v<typename T::AccDataType>;

// Requirements for struct that defines the layout used in the GEMM operation.
//
// Example that satisfies this constraint:
// struct Layouts {
//     using ALayout = RowMajor;
//     using BLayout = ColMajor;
//     using CLayout = RowMajor;
// };
template <typename T>
concept DefinesGemmLayout = requires {
    typename T::ALayout;
    typename T::BLayout;
    typename T::CLayout;
};

// A dummy placeholder for a real GEMM.
class Gemm
{
    public:
    void run([[maybe_unused]] GemmHostArgs args) const
    {
        std::cout << "Running fake GEMM" << std::endl;
    }
};

// A minimal GEMM builder, this is where all the work will be.
template <DefinesGemmTypes Types, DefinesGemmLayout Layout>
class GemmBuilder
{
    public:
    using value       = Gemm;
    using types_type  = Types;
    using layout_type = Layout;
};

} // namespace ck_tile::builder
