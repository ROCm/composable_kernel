// SPDX-License-Identifier: MIT
// Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

using Bilinear = ck::tensor_operation::element_wise::Bilinear;

namespace ck {
namespace tensor_operation {
namespace host {

// hardcoded for NumDimM == NumDimN == NumDimK == 2
template <ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          bool UseDToBinaryOp,
          typename DDataType                                                  = float,
          ck::enable_if_t<NumDimM == 2 && NumDimN == 2 && NumDimK == 2, bool> = false>
struct ReferenceContraction_M2_N2_K2 : public ck::tensor_operation::device::BaseOperator
{
    // Argument
    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_ms_ks,
                 const Tensor<BDataType>& b_ns_ks,
                 const Tensor<DDataType>& d_ms_ns,
                 Tensor<EDataType>& e_ms_ns,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : a_ms_ks_{a_ms_ks},
              b_ns_ks_{b_ns_ks},
              d_ms_ns_{d_ms_ns},
              e_ms_ns_{e_ms_ns},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
        }

        const Tensor<ADataType>& a_ms_ks_;
        const Tensor<BDataType>& b_ns_ks_;
        const Tensor<DDataType>& d_ms_ns_;
        Tensor<EDataType>& e_ms_ns_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public ck::tensor_operation::device::BaseInvoker
    {
        using Argument = ReferenceContraction_M2_N2_K2::Argument;

        void apply_unary_op(const CDEElementwiseOperation& op, EDataType& v_e, AccDataType& v_acc)
        {
            op(v_e, v_acc);
        }

        void apply_binary_op(const CDEElementwiseOperation& op,
                             EDataType& v_e,
                             AccDataType& v_acc,
                             DDataType& v_d)
        {
            op(v_e, v_acc, v_d);
        }

        float Run(const Argument& arg)
        {
            auto f_ms_ns = [&](auto m0, auto m1, auto n0, auto n1) {
                const ck::index_t K0 = arg.a_ms_ks_.mDesc.GetLengths()[2];
                const ck::index_t K1 = arg.a_ms_ks_.mDesc.GetLengths()[3];

                AccDataType v_acc = 0;

                for(ck::index_t k0 = 0; k0 < K0; ++k0)
                {
                    for(ck::index_t k1 = 0; k1 < K1; ++k1)
                    {
                        AccDataType v_a;
                        AccDataType v_b;

                        arg.a_element_op_(
                            v_a, ck::type_convert<const AccDataType>(arg.a_ms_ks_(m0, m1, k0, k1)));
                        arg.b_element_op_(
                            v_b, ck::type_convert<const AccDataType>(arg.b_ns_ks_(n0, n1, k0, k1)));

                        v_acc += v_a * v_b;
                    }
                }

                AccDataType v_e;
                DDataType v_d =
                    arg.d_ms_ns_.GetNumOfDimension() == 0 ? 0 : arg.d_ms_ns_(m0, m1, n0, n1);
                if constexpr(UseDToBinaryOp)
                {
                    apply_binary_op(arg.cde_element_op_, v_e, v_acc, v_d);
                }
                else
                {
                    apply_unary_op(arg.cde_element_op_, v_e, v_acc);
                }

                arg.e_ms_ns_(m0, m1, n0, n1) = v_e;
            };

            make_ParallelTensorFunctor(f_ms_ns,
                                       arg.e_ms_ns_.mDesc.GetLengths()[0],
                                       arg.e_ms_ns_.mDesc.GetLengths()[1],
                                       arg.e_ms_ns_.mDesc.GetLengths()[2],
                                       arg.e_ms_ns_.mDesc.GetLengths()[3])(
                std::thread::hardware_concurrency());

            return 0;
        }

        float Run(const ck::tensor_operation::device::BaseArgument* p_arg,
                  const StreamConfig& /* stream_config */ = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg));
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    bool IsSupportedArgument(const ck::tensor_operation::device::BaseArgument*) override
    {
        return true;
    }

    static auto MakeArgument(const Tensor<ADataType>& a_ms_ks,
                             const Tensor<BDataType>& b_ns_ks,
                             const Tensor<DDataType>& d_ms_ns,
                             Tensor<EDataType>& e_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{
            a_ms_ks, b_ns_ks, d_ms_ns, e_ms_ns, a_element_op, b_element_op, cde_element_op};
    }

    static auto MakeArgument(const Tensor<ADataType>& a_ms_ks,
                             const Tensor<BDataType>& b_ns_ks,
                             Tensor<EDataType>& e_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{a_ms_ks, b_ns_ks, e_ms_ns, a_element_op, b_element_op, cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<ck::tensor_operation::device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceContraction_M2_N2_K2"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
