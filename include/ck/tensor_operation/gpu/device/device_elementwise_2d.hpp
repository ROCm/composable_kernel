#pragma once
#include <iostream>
#include <vector>

#include "device.hpp"
#include "device_elementwise.hpp"
#include "gridwise_elementwise_1d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ComputeDataType,
          typename ElementwiseFunctor,
          index_t ThreadPerBlock,
          index_t ThreadTileSize,
          index_t ScalarPerVector>
struct DeviceElementwise_2D : public DeviceElementwise<ElementwiseFunctor>
{
    static_assert(ThreadTileSize % ScalarPerVector == 0);
    static constexpr int BlockTileSize = ThreadPerBlock * ThreadTileSize;
    static constexpr auto I0           = Number<0>{};

    static auto MakeDescriptor_M0(const std::vector<int>& shape, const std::vector<int>& stride)
    {
        const int m = shape[0];
        const int n = shape[1];

        // 2d desc - [m, n]
        const auto desc_m_n =
            make_naive_tensor_descriptor(make_tuple(m, n), make_tuple(stride[0], stride[1]));

        // 1d desc - [m * n]
        return transform_tensor_descriptor(desc_m_n,
                                           make_tuple(make_merge_transform(make_tuple(m, n))),
                                           make_tuple(Sequence<0, 1>{}),
                                           make_tuple(Sequence<0>{}));
    }

    using GridDesc_M0     = decltype(MakeDescriptor_M0({1, 1}, {1, 1}));
    using GridwiseEltwise = GridwiseElementwise_1D<ADataType,
                                                   BDataType,
                                                   CDataType,
                                                   ComputeDataType,
                                                   GridDesc_M0,
                                                   ElementwiseFunctor,
                                                   ThreadPerBlock,
                                                   ThreadTileSize,
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
              a_grid_desc_m0_(MakeDescriptor_M0(shape, stride_a)),
              b_grid_desc_m0_(MakeDescriptor_M0(shape, stride_b)),
              c_grid_desc_m0_(MakeDescriptor_M0(shape, stride_c)),
              functor_(functor)
        {
        }

        const ADataType* p_a_;
        const BDataType* p_b_;
        CDataType* p_c_;
        GridDesc_M0 a_grid_desc_m0_;
        GridDesc_M0 b_grid_desc_m0_;
        GridDesc_M0 c_grid_desc_m0_;
        ElementwiseFunctor functor_;
    };

    struct Invoker : public BaseInvoker
    {
        index_t CalculateGridSize(const GridDesc_M0& grid_desc_m0)
        {
            const auto gridTileSize = grid_desc_m0.GetLength(I0);
            return gridTileSize / BlockTileSize;
        }

        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto kernel      = kernel_elementwise_1d<GridwiseEltwise,
                                                      ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      GridDesc_M0,
                                                      ElementwiseFunctor>;
            float avgTime          = 0;
            const index_t gridSize = CalculateGridSize(arg.c_grid_desc_m0_);
            if(nrepeat == 0)
            {
                launch_kernel(kernel,
                              dim3(gridSize),
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
                                                 dim3(gridSize),
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

        if(m0 % BlockTileSize != 0)
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
        str << "DeviceElementwise_2D"
            << "<"
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
