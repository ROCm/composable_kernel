// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/utility/functional4.hpp"
#include "ck/utility/tuple_helper.hpp"

#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

#if __clang_major__ >= 23
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
#endif
namespace ck {
namespace tensor_operation {
namespace host {

template <typename AsTensorTuple,
          typename BsTensorTuple,
          typename DsTensorTuple,
          typename EDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AComputeType,
          typename BComputeType>
struct ReferenceGemmMultiABD : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const AsTensorTuple& as_m_k,
                 const BsTensorTuple& bs_k_n,
                 const DsTensorTuple& ds_m_n,
                 Tensor<EDataType>& e_m_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : as_m_k_{as_m_k},
              bs_k_n_{bs_k_n},
              ds_m_n_{ds_m_n},
              e_m_n_{e_m_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
        }

        const AsTensorTuple& as_m_k_;
        const BsTensorTuple& bs_k_n_;
        const DsTensorTuple& ds_m_n_;
        Tensor<EDataType>& e_m_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceGemmMultiABD::Argument;

        float Run(const Argument& arg)
        {
            static constexpr index_t NumATensor = AsTensorTuple::Size();
            static constexpr index_t NumBTensor = BsTensorTuple::Size();
            static constexpr index_t NumDTensor = DsTensorTuple::Size();

            const int M = arg.as_m_k_[Number<0>{}].mDesc.GetLengths()[0];
            const int K = arg.as_m_k_[Number<0>{}].mDesc.GetLengths()[1];
            const int N = arg.bs_k_n_[Number<0>{}].mDesc.GetLengths()[1];

            Tensor<AComputeType> a_m_k({M, K});
            for(int m = 0; m < M; ++m)
            {
                for(int k = 0; k < K; ++k)
                {
                    // result
                    auto data_refs1 = ck::tie(a_m_k(m, k));
                    // inputs
                    auto data_refs2 = generate_tie(
                        [&](auto i) -> auto& { return arg.as_m_k_[Number<i>{}](m, k); },
                        Number<NumATensor>{});
                    auto data_refs = concat_tuple_of_reference(data_refs1, data_refs2);
                    unpack(arg.a_element_op_, data_refs);
                }
            }

            Tensor<BComputeType> b_k_n({K, N});
            for(int k = 0; k < K; ++k)
            {
                for(int n = 0; n < N; ++n)
                {
                    // result
                    auto data_refs1 = ck::tie(b_k_n(k, n));
                    // inputs
                    auto data_refs2 = generate_tie(
                        [&](auto i) -> auto& { return arg.bs_k_n_[Number<i>{}](k, n); },
                        Number<NumBTensor>{});
                    auto data_refs = concat_tuple_of_reference(data_refs1, data_refs2);
                    unpack(arg.b_element_op_, data_refs);
                }
            }

            using PassThrough = ck::tensor_operation::element_wise::PassThrough;
            Tensor<AccDataType> c_m_n({M, N});

            using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<AComputeType,
                                                                                    BComputeType,
                                                                                    AccDataType,
                                                                                    AccDataType,
                                                                                    PassThrough,
                                                                                    PassThrough,
                                                                                    PassThrough>;
            auto ref_gemm               = ReferenceGemmInstance{};
            auto ref_invoker            = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(
                a_m_k, b_k_n, c_m_n, PassThrough{}, PassThrough{}, PassThrough{});

            ref_invoker.Run(ref_argument);

            for(int m = 0; m < M; ++m)
            {
                for(int n = 0; n < N; ++n)
                {
                    // compulsory
                    auto data_refs1 = ck::tie(arg.e_m_n_(m, n), c_m_n(m, n));
                    // optional (if multiple Ds)
                    auto data_refs2 = generate_tie(
                        [&](auto i) -> auto& { return arg.ds_m_n_[Number<i>{}](m, n); },
                        Number<NumDTensor>{});
                    auto data_refs = concat_tuple_of_reference(data_refs1, data_refs2);
                    unpack(arg.cde_element_op_, data_refs);
                }
            }

            return 0;
        }

        float Run(const device::BaseArgument* p_arg,
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

    bool IsSupportedArgument(const device::BaseArgument*) override { return true; }

    static auto MakeArgument(const AsTensorTuple& as_m_k,
                             const BsTensorTuple& bs_k_n,
                             const DsTensorTuple& ds_m_n,
                             Tensor<EDataType>& e_m_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{as_m_k, bs_k_n, ds_m_n, e_m_n, a_element_op, b_element_op, cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    virtual std::unique_ptr<device::BaseInvoker> MakeInvokerPointer()
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "ReferenceGemmMultiABD"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck

#if __clang_major__ >= 23
#pragma clang diagnostic pop
#endif
