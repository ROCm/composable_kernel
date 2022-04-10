#pragma once
#include <iostream>
#include <vector>

#include "device.hpp"
#include "device_elementwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          index_t BlockSize,
          typename ElementwiseFunctor>
struct DeviceElementwise_2D : public DeviceElementwise<ElementwiseFunctor>
{
    static auto Make2dDescriptor_M_N(const std::vector<int>& shape, const std::vector<int>& stride)
    {
        return make_naive_tensor_descriptor(make_tuple(shape[0], shape[1]),
                                            make_tuple(stride[0], stride[1]));
    }

    using GridDesc_M_N = decltype(Make2dDescriptor_M_N({1, 1}, {1, 1}));

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
        float Run(const Argument& arg, int nrepeat = 1)
        {
            // TODO
            (void)arg;
            (void)nrepeat;
            return 0;
        }

        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        // TODO: properly implement this check
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);
        return pArg != nullptr;
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
            << BlockSize
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
