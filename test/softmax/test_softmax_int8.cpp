#include "gtest/gtest.h"
#include "test_softmax_util.hpp"

template <ck::index_t N>
using I = ck::Number<N>;

template <typename Tuple>
class TestSoftmaxINT8Reduce1Dim : public ck::TestSoftmax<Tuple>
{
};

template <typename Tuple>
class TestSoftmaxINT8Reduce1DInnerDims : public ck::TestSoftmax<Tuple>
{
};

template <typename Tuple>
class TestSoftmaxINT8Reduce2Dims : public ck::TestSoftmax<Tuple>
{
};

template <typename Tuple>
class TestSoftmaxINT8Reduce2DInnerDims : public ck::TestSoftmax<Tuple>
{
};

template <typename Tuple>
class TestSoftmaxINT8ReduceAllDims : public ck::TestSoftmax<Tuple>
{
};

// clang-format off
using KernelTypesReduce1Dim = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<4>, I<64>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<64>, I<1>, I<16>, I<16>>
    >;

using KernelTypesReduce1DInnerDims = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<16>, I<1>, I<1>, I<1>>,  // fallback
    std::tuple<int8_t, float, int8_t, I<3>, I<1>, I<256>, I<8>, I<32>, I<8>, I<8>, I<0>, I<8>, I<8>>
    >;

using KernelTypesReduce2Dims = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<8>, I<32>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<4>, I<64>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<2>, I<128>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<64>, I<1>, I<16>, I<16>>
    >;

using KernelTypesReduce2DInnerDims = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<2>, I<128>, I<1>, I<16>, I<1>, I<1>, I<1>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<2>, I<256>, I<8>, I<32>, I<8>, I<8>, I<0>, I<8>, I<8>>
    >;

using KernelTypesReduceAllDims = ::testing::Types<
// InDataType, AccDataType, OutDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<8>, I<32>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<4>, I<64>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<2>, I<128>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<16>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<32>, I<1>, I<16>, I<16>>,
    std::tuple<int8_t, float, int8_t, I<3>, I<3>, I<256>, I<1>, I<256>, I<1>, I<64>, I<1>, I<16>, I<16>>
    >;
// clang-format on

TYPED_TEST_SUITE(TestSoftmaxINT8Reduce1Dim, KernelTypesReduce1Dim);
TYPED_TEST_SUITE(TestSoftmaxINT8Reduce1DInnerDims, KernelTypesReduce1DInnerDims);
TYPED_TEST_SUITE(TestSoftmaxINT8Reduce2Dims, KernelTypesReduce2Dims);
TYPED_TEST_SUITE(TestSoftmaxINT8Reduce2DInnerDims, KernelTypesReduce2DInnerDims);
TYPED_TEST_SUITE(TestSoftmaxINT8ReduceAllDims, KernelTypesReduceAllDims);

TYPED_TEST(TestSoftmaxINT8Reduce1Dim, ReduceInnermostDim)
{
    std::vector<ck::index_t> reduce_dims{2};
    this->Run(reduce_dims);
}
TYPED_TEST(TestSoftmaxINT8Reduce1Dim, IncorrectReduceDims)
{
    std::vector<ck::index_t> reduce_dims{-1};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
    reduce_dims = std::vector<ck::index_t>{3};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
TYPED_TEST(TestSoftmaxINT8Reduce1Dim, ReduceMoreDims)
{
    std::vector<ck::index_t> reduce_dims{1, 2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
    reduce_dims = std::vector<ck::index_t>{0, 1, 2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
TYPED_TEST(TestSoftmaxINT8Reduce1DInnerDims, ReduceInnerDims)
{
    std::vector<ck::index_t> reduce_dims{1};
    this->Run(reduce_dims);
    reduce_dims = std::vector<ck::index_t>{0};
    this->Run(reduce_dims);
}
TYPED_TEST(TestSoftmaxINT8Reduce2Dims, Reduce2Dims)
{
    std::vector<ck::index_t> reduce_dims{1, 2};
    this->Run(reduce_dims);
    reduce_dims = std::vector<ck::index_t>{0, 2};
    this->Run(reduce_dims);
}
TYPED_TEST(TestSoftmaxINT8Reduce2Dims, ReduceTooLessDims)
{
    std::vector<ck::index_t> reduce_dims{2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
    reduce_dims = std::vector<ck::index_t>{1};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
TYPED_TEST(TestSoftmaxINT8Reduce2Dims, ReduceIncorrectDims)
{
    std::vector<ck::index_t> reduce_dims{2, 4};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
    reduce_dims = std::vector<ck::index_t>{2, -1};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
TYPED_TEST(TestSoftmaxINT8Reduce2Dims, ReduceTooManyDims)
{
    std::vector<ck::index_t> reduce_dims{0, 1, 2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
TYPED_TEST(TestSoftmaxINT8Reduce2DInnerDims, ReduceInnerDims)
{
    std::vector<ck::index_t> reduce_dims{0, 1};
    this->Run(reduce_dims);
}
TYPED_TEST(TestSoftmaxINT8ReduceAllDims, ReduceAllDims)
{
    std::vector<ck::index_t> reduce_dims{0, 1, 2};
    this->Run(reduce_dims);
}
TYPED_TEST(TestSoftmaxINT8ReduceAllDims, ReduceTooLessDims)
{
    std::vector<ck::index_t> reduce_dims{2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
    reduce_dims = std::vector<ck::index_t>{1, 2};
    EXPECT_THROW(this->Run(reduce_dims), std::runtime_error);
}
