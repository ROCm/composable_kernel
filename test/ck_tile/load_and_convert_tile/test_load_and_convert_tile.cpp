// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/common.hpp"
#include "kernel.hpp"

// Helper to print matrix (for debugging)
template <typename T>
void print_matrix(const ck_tile::HostTensor<T>& mat, const std::string& name = "Matrix")
{
    const auto lens = mat.get_lengths();
    assert(len(lens) == 2);
    const ck_tile::index_t rows  = lens[0];
    const ck_tile::index_t cols  = lens[1];
    const ck_tile::index_t limit = 10;

    std::cout << name << " (" << rows << "×" << cols << "):\n";
    for(ck_tile::index_t i = 0; i < std::min(rows, ck_tile::index_t(limit)); ++i)
    {
        for(ck_tile::index_t j = 0; j < std::min(cols, ck_tile::index_t(limit)); ++j)
        {
            std::cout << std::setw(3) << std::setprecision(3)
                      << ck_tile::type_convert<float>(mat(i, j)) << " ";
        }
        if(cols > limit)
            std::cout << "...";
        std::cout << "\n";
    }
    if(rows > limit)
        std::cout << "...\n";
    std::cout << "\n";
}

template <typename Tuple>
class TestLoadAndConvert : public ::testing::Test
{
    public:
    using XDataType     = std::tuple_element_t<0, Tuple>;
    using YDataType     = std::tuple_element_t<1, Tuple>;
    using LoadTranspose = std::tuple_element_t<2, Tuple>;

    protected:
    void RunTest()
    {
        constexpr ck_tile::index_t M = 64;
        constexpr ck_tile::index_t N = 64;
        constexpr ck_tile::index_t K = 32;

        ck_tile::HostTensor<XDataType> h_a({M, K});
        ck_tile::HostTensor<YDataType> h_c({M, K});
        ck_tile::FillUniformDistributionIntegerValue<XDataType>{-5.0, 5.0, 11939}(h_a);

        ck_tile::DeviceMem d_a(h_a.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d_c(h_c.get_element_space_size_in_bytes());

        d_a.ToDevice(h_a.data());
        d_c.ToDevice(h_c.data());

        using BlockWarps = ck_tile::sequence<1, 1, 1>;
        using BlockTile  = ck_tile::sequence<32, 32, 16>;
        using WarpTile   = ck_tile::sequence<32, 32, 16>;
        using Vector     = ck_tile::sequence<1, 8>;

        using Shape   = ck_tile::LoadAndConvertShape<BlockWarps, BlockTile, WarpTile, Vector>;
        using Problem = ck_tile::LoadAndConvertProblem<XDataType, YDataType, Shape, LoadTranspose>;
        using Kernel  = ck_tile::LoadAndConvertKernel<Problem>;

        constexpr ck_tile::index_t block_size = Kernel::kBlockSize;
        const ck_tile::index_t grid_size      = ck_tile::integer_divide_ceil(M, Shape::Block_M) *
                                           ck_tile::integer_divide_ceil(N, Shape::Block_N);

        launch_kernel(ck_tile::stream_config{nullptr, true},
                      make_kernel<block_size>(Kernel{},
                                              dim3(grid_size),
                                              dim3(block_size),
                                              0,
                                              static_cast<const XDataType*>(d_a.GetDeviceBuffer()),
                                              static_cast<YDataType*>(d_c.GetDeviceBuffer()),
                                              M,
                                              N,
                                              K));

        ck_tile::hip_check_error(hipDeviceSynchronize());
        d_c.FromDevice(h_c.data());
        ck_tile::HostTensor<YDataType> h_a_ref({M, K});
        ck_tile::reference_unary_elementwise<XDataType, YDataType, float>(
            h_a, h_a_ref, [](const auto& x) { return x; });
        bool pass = ck_tile::check_err(h_c, h_a_ref);

        // print_matrix(h_a, "Matrix A");
        // print_matrix(h_c, "Matrix C");

        EXPECT_TRUE(pass);
    }
};

using TestTypes = ::testing::Types<std::tuple<ck_tile::half_t, ck_tile::half_t, std::false_type>,
                                   // std::tuple<ck_tile::half_t, ck_tile::fp8_t, std::false_type>,
                                   // std::tuple<ck_tile::fp8_t, ck_tile::half_t, std::false_type>,
                                   std::tuple<ck_tile::fp8_t, ck_tile::fp8_t, std::false_type>,
                                   std::tuple<ck_tile::half_t, ck_tile::half_t, std::true_type>,
                                   // std::tuple<ck_tile::half_t, ck_tile::fp8_t, std::true_type>,
                                   // std::tuple<ck_tile::fp8_t, ck_tile::half_t, std::true_type>,
                                   std::tuple<ck_tile::fp8_t, ck_tile::fp8_t,  std::true_type>>;

TYPED_TEST_SUITE(TestLoadAndConvert, TestTypes);

TYPED_TEST(TestLoadAndConvert, Test) { this->RunTest(); }
