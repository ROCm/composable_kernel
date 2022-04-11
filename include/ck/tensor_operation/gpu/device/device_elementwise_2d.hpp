#pragma once
#include <iostream>
#include <vector>

#include "device.hpp"
#include "device_elementwise.hpp"
#include "gridwise_elementwise_2d.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ElementwiseFunctor,
          index_t MThreadPerBlock,
          index_t NThreadPerBlock,
          index_t MThreadTileSize,
          index_t NThreadTileSize>
struct DeviceElementwise_2D : public DeviceElementwise<ElementwiseFunctor>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static auto Make2dDescriptor_M_N(const std::vector<int>& shape, const std::vector<int>& stride)
    {
        return make_naive_tensor_descriptor(make_tuple(shape[0], shape[1]),
                                            make_tuple(stride[0], stride[1]));
    }

    static constexpr index_t BlockSize   = MThreadPerBlock * NThreadPerBlock;
    static constexpr int M_BlockTileSize = MThreadPerBlock * MThreadTileSize;
    static constexpr int N_BlockTileSize = NThreadPerBlock * NThreadTileSize;

    using GridDesc_M_N    = decltype(Make2dDescriptor_M_N({1, 1}, {1, 1}));
    using GridwiseEltwise = GridwiseElementwise_2D<ADataType,
                                                   BDataType,
                                                   CDataType,
                                                   GridDesc_M_N,
                                                   GridDesc_M_N,
                                                   GridDesc_M_N,
                                                   ElementwiseFunctor,
                                                   MThreadTileSize,
                                                   NThreadTileSize>;

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
              a_grid_desc_m_n_(Make2dDescriptor_M_N(shape, stride_a)),
              b_grid_desc_m_n_(Make2dDescriptor_M_N(shape, stride_b)),
              c_grid_desc_m_n_(Make2dDescriptor_M_N(shape, stride_c)),
              functor_(functor)
        {
        }

        const ADataType* p_a_;
        const BDataType* p_b_;
        CDataType* p_c_;
        GridDesc_M_N a_grid_desc_m_n_;
        GridDesc_M_N b_grid_desc_m_n_;
        GridDesc_M_N c_grid_desc_m_n_;
        ElementwiseFunctor functor_;
    };

    struct Invoker : public BaseInvoker
    {
        index_t CalculateGridSize(const GridDesc_M_N& grid_desc_m_n)
        {
            const auto M = grid_desc_m_n.GetLength(I0);
            const auto N = grid_desc_m_n.GetLength(I1);

            assert(M % M_BlockTileSize == 0);
            assert(N % N_BlockTileSize == 0);

            return (M / M_BlockTileSize) * (N / N_BlockTileSize);
        }

        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto kernel = kernel_elementwise_2d<GridwiseEltwise,
                                                      ADataType,
                                                      BDataType,
                                                      CDataType,
                                                      GridDesc_M_N,
                                                      GridDesc_M_N,
                                                      GridDesc_M_N,
                                                      ElementwiseFunctor>;
            // TODO
            (void)arg;
            (void)nrepeat;
            (void)kernel;
            float avgTime          = 0;
            const index_t gridSize = CalculateGridSize(arg.c_grid_desc_m_n_);
            if(nrepeat == 0)
            {
                launch_kernel(kernel,
                              dim3(gridSize),
                              dim3(BlockSize),
                              0,
                              arg.p_a_,
                              arg.p_b_,
                              arg.p_c_,
                              arg.a_grid_desc_m_n_,
                              arg.b_grid_desc_m_n_,
                              arg.c_grid_desc_m_n_,
                              arg.functor_);
            }
            else
            {
                avgTime = launch_and_time_kernel(kernel,
                                                 nrepeat,
                                                 dim3(gridSize),
                                                 dim3(BlockSize),
                                                 0,
                                                 arg.p_a_,
                                                 arg.p_b_,
                                                 arg.p_c_,
                                                 arg.a_grid_desc_m_n_,
                                                 arg.b_grid_desc_m_n_,
                                                 arg.c_grid_desc_m_n_,
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

        const auto M = pArg->c_grid_desc_m_n_.GetLength(I0);
        const auto N = pArg->c_grid_desc_m_n_.GetLength(I1);

        if(M % M_BlockTileSize != 0 && N % N_BlockTileSize != 0)
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
