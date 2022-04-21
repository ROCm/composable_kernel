#pragma once
#include <iostream>
#include <vector>

#include "device.hpp"
#include "device_binary_elementwise.hpp"
#include "gridwise_binary_elementwise_1d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ComputeDataType,
          typename ElementwiseFunctor,
          index_t ThreadPerBlock,
          index_t ScalarPerVector>
struct DeviceBinaryElementwise_2D : public DeviceBinaryElementwise<ElementwiseFunctor>
{
    static constexpr auto I0 = Number<0>{};

    static auto MakeDescriptor_M0(const std::vector<int>& shape,
                                  const std::vector<int>& stride,
                                  index_t gridSize)
    {
        const int m = shape[0];
        const int n = shape[1];

        // 2d desc - [m, n]
        const auto desc_m_n =
            make_naive_tensor_descriptor(make_tuple(m, n), make_tuple(stride[0], stride[1]));

        // 1d desc - [m * n]
        const auto desc_m0 =
            transform_tensor_descriptor(desc_m_n,
                                        make_tuple(make_merge_transform(make_tuple(m, n))),
                                        make_tuple(Sequence<0, 1>{}),
                                        make_tuple(Sequence<0>{}));

        // pad
        const auto m0           = desc_m0.GetLength(I0);
        const index_t loop_step = gridSize * ThreadPerBlock * ScalarPerVector;
        const auto pad          = math::integer_least_multiple(m0, loop_step) - m0;
        const auto desc_m0_pad =
            transform_tensor_descriptor(desc_m0,
                                        make_tuple(make_right_pad_transform(m0, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m0_pad;
    }

    using GridDesc_M0 = decltype(MakeDescriptor_M0({1, 1}, {1, 1}, 1));
    using GridwiseBinEltwise = GridwiseBinaryElementwise_1D<ADataType,
                                                            BDataType,
                                                            CDataType,
                                                            ComputeDataType,
                                                            GridDesc_M0,
                                                            ElementwiseFunctor,
                                                            ScalarPerVector>;

    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a,
                 const BDataType* p_b,
                 CDataType* p_c,
                 const std::vector<int>& shape,
                 const std::vector<int>& stride_a,
                 const std::vector<int>& stride_b,
                 const std::vector<int>& stride_c,
                 ElementwiseFunctor functor)
            : p_a_(p_a),
              p_b_(p_b),
              p_c_(p_c),
              functor_(functor),
              gridSize_(128) // FIXME - Calculate the grid size by number of CU in the future
        {
            a_grid_desc_m0_ = MakeDescriptor_M0(shape, stride_a, gridSize_);
            b_grid_desc_m0_ = MakeDescriptor_M0(shape, stride_b, gridSize_);
            c_grid_desc_m0_ = MakeDescriptor_M0(shape, stride_c, gridSize_);
        }

        const ADataType* p_a_;
        const BDataType* p_b_;
        CDataType* p_c_;
        GridDesc_M0 a_grid_desc_m0_;
        GridDesc_M0 b_grid_desc_m0_;
        GridDesc_M0 c_grid_desc_m0_;
        ElementwiseFunctor functor_;
        index_t gridSize_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            (void)arg;
            const auto kernel = kernel_elementwise_1d<GridwiseBinEltwise,
                                                      ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      GridDesc_M0,
                                                      ElementwiseFunctor>;
            float avgTime = 0;
            if(nrepeat == 0)
            {
                launch_kernel(kernel,
                              dim3(arg.gridSize_),
                              dim3(ThreadPerBlock),
                              0,
                              arg.p_a_,
                              arg.p_b_,
                              arg.p_c_,
                              arg.a_grid_desc_m0_,
                              arg.b_grid_desc_m0_,
                              arg.c_grid_desc_m0_,
                              arg.functor_);
            }
            else
            {
                avgTime = launch_and_time_kernel(kernel,
                                                 nrepeat,
                                                 dim3(arg.gridSize_),
                                                 dim3(ThreadPerBlock),
                                                 0,
                                                 arg.p_a_,
                                                 arg.p_b_,
                                                 arg.p_c_,
                                                 arg.a_grid_desc_m0_,
                                                 arg.b_grid_desc_m0_,
                                                 arg.c_grid_desc_m0_,
                                                 arg.functor_);
            }
            return avgTime;
        }

        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if(pArg == nullptr)
            return false;

        // m * n
        const auto m0 = pArg->c_grid_desc_m0_.GetLength(I0);

        if(m0 % ScalarPerVector != 0)
            return false;

        return true;
    };

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      const std::vector<int>& shape,
                                                      const std::vector<int>& stride_a,
                                                      const std::vector<int>& stride_b,
                                                      const std::vector<int>& stride_c,
                                                      ElementwiseFunctor functor) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          shape,
                                          stride_a,
                                          stride_b,
                                          stride_c,
                                          functor);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBinaryElementwise_2D"
            << "<"
            << "ThreadPerBlock = " << ThreadPerBlock
            << "ScalarPerVector = " << ScalarPerVector
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
