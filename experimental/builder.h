#pragma once
#include <iostream>
#include <concepts>

namespace ck_tile::builder {

using index_t = std::size_t;

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

struct ColMajor
{
};
struct RowMajor
{
};

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

template <typename T>
concept DefinesGemmLayout = requires {
    typename T::ALayout;
    typename T::BLayout;
    typename T::CLayout;
};

class Gemm
{
    public:
    void run([[maybe_unused]] GemmHostArgs args) const
    {
        std::cout << "Running fake GEMM" << std::endl;
    }
};

template <DefinesGemmTypes Types, DefinesGemmLayout Layout>
class GemmBuilder
{
    public:
    using value       = Gemm;
    using types_type  = Types;
    using layout_type = Layout;
};

} // namespace ck_tile::builder
