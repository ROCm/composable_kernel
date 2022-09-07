#include "gtest/gtest.h"
#include "test_softmax_util.hpp"

template <ck::index_t N>
using I = ck::Number<N>;

template <typename Tuple>
class TestSoftmaxINT8 : public ck::TestSoftmax<Tuple>
{
};

// clang-format off
using KernelTypes = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<4>, I<64>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<64>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<8>, I<32>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<4>, I<64>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<2>, I<128>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<64>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<8>, I<32>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<4>, I<64>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<2>, I<128>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<64>, I<1>, I<16>, I<16>>
    >;
// clang-format on
TYPED_TEST_SUITE(TestSoftmaxINT8, KernelTypes);
TYPED_TEST(TestSoftmaxINT8, Test_INT8) { this->Run(); }

template <typename Tuple>
class TestSoftmaxINT8v2 : public ck::TestSoftmax<Tuple>
{
};

// clang-format off
using KernelTypesReduceSingleDim = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<4>, I<64>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<64>, I<1>, I<16>, I<16>>
    >;
// clang-format on
TYPED_TEST_SUITE(TestSoftmaxINT8v2, KernelTypesReduceSingleDim);

TYPED_TEST(TestSoftmaxINT8v2, Test_INT8_reduce_inner_dim)
{
    std::vector<ck::index_t> reduce_dims{1};
    this->Run(reduce_dims);
}

TYPED_TEST(TestSoftmaxINT8v2, Test_INT8_reduce_outermost_dim)
{
    std::vector<ck::index_t> reduce_dims{0};
    this->Run(reduce_dims);
}
