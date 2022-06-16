#include <vector>
#include <iostream>
#include <half.hpp>
#include "gtest/gtest.h"

#include "config.hpp"
#include "host_tensor.hpp"
#include "check_err.hpp"
#include "number.hpp"
#include "reference_softmax.hpp"
#include "device_softmax.hpp"

using namespace ck;

template <index_t N>
using I = ck::Number<N>;

template <typename Tuple>
class TestSoftmax : public ::testing::Test
{
    protected:
    using InDataType                            = std::tuple_element_t<0, Tuple>;
    using AccDataType                           = std::tuple_element_t<1, Tuple>;
    using OutDataType                           = std::tuple_element_t<2, Tuple>;
    using ScalarDataType                        = std::tuple_element_t<3, Tuple>;
    static constexpr index_t Rank               = std::tuple_element_t<4, Tuple>{}.value;
    static constexpr index_t NumReduceDim       = std::tuple_element_t<5, Tuple>{}.value;
    static constexpr index_t BlockSize          = std::tuple_element_t<6, Tuple>{}.value;
    static constexpr index_t MThreadClusterSize = std::tuple_element_t<7, Tuple>{}.value;
    static constexpr index_t KThreadClusterSize = std::tuple_element_t<8, Tuple>{}.value;
    static constexpr index_t MThreadSliceSize   = std::tuple_element_t<9, Tuple>{}.value;
    static constexpr index_t KThreadSliceSize   = std::tuple_element_t<10, Tuple>{}.value;
    static constexpr index_t InSrcVectorDim     = std::tuple_element_t<11, Tuple>{}.value;
    static constexpr index_t InSrcVectorSize    = std::tuple_element_t<12, Tuple>{}.value;
    static constexpr index_t OutDstVectorSize   = std::tuple_element_t<13, Tuple>{}.value;

    using ReferenceInstance = tensor_operation::host::
        ReferenceSoftmax<InDataType, OutDataType, AccDataType, ScalarDataType>;

    using DeviceInstance = tensor_operation::device::DeviceSoftmax<InDataType,
                                                                   AccDataType,
                                                                   OutDataType,
                                                                   ScalarDataType,
                                                                   Rank,
                                                                   NumReduceDim,
                                                                   BlockSize,
                                                                   MThreadClusterSize,
                                                                   KThreadClusterSize,
                                                                   MThreadSliceSize,
                                                                   KThreadSliceSize,
                                                                   InSrcVectorDim,
                                                                   InSrcVectorSize,
                                                                   OutDstVectorSize>;

    TestSoftmax() : ref_instance_invoker_(ReferenceInstance{}.MakeInvoker()) {}

    void RunSingle(std::vector<index_t> in_length, ScalarDataType alpha, ScalarDataType beta)
    {
        std::vector<index_t> reduce_dims(NumReduceDim);
        std::iota(reduce_dims.begin(), reduce_dims.end(), Rank - NumReduceDim);

        Tensor<InDataType> in(in_length);
        Tensor<OutDataType> out(in_length);

        in.GenerateTensorValue(GeneratorTensor_2<InDataType>{-5, 5});
        out.GenerateTensorValue(GeneratorTensor_2<OutDataType>{-5, 5});

        Tensor<OutDataType> out_ref(out);

        DeviceMem in_dev(sizeof(InDataType) * in.mDesc.GetElementSpace());
        DeviceMem out_dev(sizeof(OutDataType) * out.mDesc.GetElementSpace());
        in_dev.ToDevice(in.mData.data());
        out_dev.ToDevice(out.mData.data());

        std::vector<index_t> i_in_lengths(in.mDesc.GetLengths().begin(),
                                          in.mDesc.GetLengths().end());
        std::vector<index_t> i_in_strides(in.mDesc.GetStrides().begin(),
                                          in.mDesc.GetStrides().end());

        auto device_instance = DeviceInstance{};
        auto argument_ptr    = device_instance.MakeArgumentPointer(i_in_lengths,
                                                                i_in_strides,
                                                                reduce_dims,
                                                                alpha,
                                                                beta,
                                                                in_dev.GetDeviceBuffer(),
                                                                out_dev.GetDeviceBuffer());

        if(!device_instance.IsSupportedArgument(argument_ptr.get()))
        {
            FAIL() << "Unsupported argument";
        }

        auto invoker_ptr = device_instance.MakeInvokerPointer();
        invoker_ptr->Run(argument_ptr.get());

        ref_instance_invoker_.Run({in, out_ref, alpha, beta, Rank, reduce_dims});

        out_dev.FromDevice(out.mData.data());
        EXPECT_TRUE(ck::utils::check_err(out.mData, out_ref.mData));
    }

    void Run()
    {
        for(auto in_length : this->in_lengths_)
        {
            for(auto scale : this->scales_)
            {
                this->RunSingle(in_length, std::get<0>(scale), std::get<1>(scale));
            }
        }
    }

    std::vector<std::vector<index_t>> in_lengths_ = {{1, 8, 128}, {2, 128, 1024}, {3, 9, 1032}};
    std::vector<std::tuple<ScalarDataType, ScalarDataType>> scales_ = {{1, 0}, {2, 2}, {0, 1}};

    typename ReferenceInstance::Invoker ref_instance_invoker_;
};

// clang-format off
using KernelTypes = ::testing::Types<
// InDataType, AccDataType, OutDataType, ScalarDataType, Rank, NumReduceDim, BlockSize, MThreadClusterSize, KThreadClusterSize, MThreadSliceSize, KThreadSliceSize, InSrcVectorDim, InSrcVectorSize, OutDstVectorSize>
    // FP16
    std::tuple<ck::half_t, float, ck::half_t, float, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, float, I<3>, I<1>, I<256>, I<4>, I<64>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, float, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, float, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, float, I<3>, I<2>, I<256>, I<8>, I<32>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, float, I<3>, I<2>, I<256>, I<4>, I<64>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, float, I<3>, I<2>, I<256>, I<2>, I<128>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    std::tuple<ck::half_t, float, ck::half_t, float, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<8>, I<1>, I<8>, I<8>>,
    // FP32
    std::tuple<float, float, float, float, I<3>, I<1>, I<256>, I<8>, I<32>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, float, I<3>, I<1>, I<256>, I<4>, I<64>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, float, I<3>, I<1>, I<256>, I<2>, I<128>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, float, I<3>, I<1>, I<256>, I<1>, I<256>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, float, I<3>, I<2>, I<256>, I<8>, I<32>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, float, I<3>, I<2>, I<256>, I<4>, I<64>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, float, I<3>, I<2>, I<256>, I<2>, I<128>, I<1>, I<4>, I<1>, I<4>, I<4>>,
    std::tuple<float, float, float, float, I<3>, I<2>, I<256>, I<1>, I<256>, I<1>, I<4>, I<1>, I<4>, I<4>>
    >;
// clang-format on
TYPED_TEST_SUITE(TestSoftmax, KernelTypes);
TYPED_TEST(TestSoftmax, Test_FP16) { this->Run(); }
