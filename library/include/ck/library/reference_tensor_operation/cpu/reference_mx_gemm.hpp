// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ScaleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ComputeTypeA = CDataType,
          typename ComputeTypeB = ComputeTypeA>
struct ReferenceMXGemm : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_m_k,
                 const Tensor<ScaleDataType>& a_m_kblock_scales,
                 const Tensor<BDataType>& b_k_n,
                 const Tensor<ScaleDataType>& b_kblock_n_scales,
                 Tensor<CDataType>& c_m_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_m_k_{a_m_k},
              a_m_kblock_scales_{a_m_kblock_scales},
              b_k_n_{b_k_n},
              b_kblock_n_scales_{b_kblock_n_scales},
              c_m_n_{c_m_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ADataType>& a_m_k_;
        const Tensor<ScaleDataType>& a_m_kblock_scales_;
        const Tensor<BDataType>& b_k_n_;
        const Tensor<ScaleDataType>& b_kblock_n_scales_;
        Tensor<CDataType>& c_m_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceMXGemm::Argument;

        float Run(const Argument& arg)
        {
            using GemmInstance = ck::tensor_operation::host::ReferenceGemm<ComputeTypeA,
                                                                           ComputeTypeB,
                                                                           CDataType,
                                                                           AccDataType,
                                                                           AElementwiseOperation,
                                                                           BElementwiseOperation,
                                                                           CElementwiseOperation,
                                                                           ComputeTypeA,
                                                                           ComputeTypeB>;

            const ck::index_t M = arg.a_m_k_.mDesc.GetLengths()[0];
            const ck::index_t N = arg.b_k_n_.mDesc.GetLengths()[1];
            assert(arg.a_m_k_.mDesc.GetLengths()[1] == arg.b_k_n_.mDesc.GetLengths()[0]);
            const ck::index_t K           = arg.a_m_k_.mDesc.GetLengths()[1];
            const ck::index_t SCALE_BLOCK = K / arg.a_m_kblock_scales_.mDesc.GetLengths()[1];
            Tensor<ComputeTypeA> a_m_k_scaled(HostTensorDescriptor({M, K}, {K, 1}));
            Tensor<ComputeTypeB> b_k_n_scaled(HostTensorDescriptor({K, N}, {1, K}));
            // printf("K: %d\n", K);

            for(int m = 0; m < M; m++)
            {
                for(int k = 0; k < K; k++)
                {
                    if constexpr(is_same_v<ADataType, f4x2_pk_t>)
                    {
                        if(k % 2 == 1)
                        {
                            continue;
                        }
                        // TODO: add support for ColMajor layout as well
                        auto a_pack = arg.a_m_k_(m, k);
                        auto a_scale =
                            type_convert<ComputeTypeA>(arg.a_m_kblock_scales_(m, k / SCALE_BLOCK));
                        auto a_f4_lo = f4_t(a_pack.template unpack<>(Number<0>{}));
                        auto a_f4_hi = f4_t(a_pack.template unpack<>(Number<1>{}));

                        a_m_k_scaled(m, k)     = type_convert<ComputeTypeA>(a_f4_lo) * a_scale;
                        a_m_k_scaled(m, k + 1) = type_convert<ComputeTypeA>(a_f4_hi) * a_scale;
                    }
                    else if constexpr(is_same_v<ADataType, f6x16_pk_t> ||
                                      is_same_v<ADataType, bf6x16_pk_t> ||
                                      is_same_v<ADataType, f6x32_pk_t> ||
                                      is_same_v<ADataType, bf6x32_pk_t>)
                    {
                        a_m_k_scaled(m, k) =
                            type_convert<ComputeTypeA>(
                                arg.a_m_k_(m, k).unpack(k % ADataType::packed_size)) *
                            type_convert<ComputeTypeA>(arg.a_m_kblock_scales_(m, k / SCALE_BLOCK));
                    }
                    else
                    {
                        a_m_k_scaled(m, k) =
                            type_convert<ComputeTypeA>(arg.a_m_k_(m, k)) *
                            type_convert<ComputeTypeA>(arg.a_m_kblock_scales_(m, k / SCALE_BLOCK));
                    }
                }
            }
#if 0
            if(K < 600)
            {
                std::cout << std::endl;
                const auto row = 4;

                std::cout << "Ref GEMM a_m_k(" << row << ",:):";
                for(int i = 0; i < K; ++i)
                {
                    if(i % (std::max(16, ck::packed_size_v<ADataType>)) == 0)
                    {
                        std::cout << std::endl << "k = " << i << " : ";
                    }

                    if constexpr(ck::is_same_v<ADataType, ck::f4x2_pk_t>)
                    {
                        if(i % 2 == 0)
                        {
                            std::cout << type_convert<float>(ck::f4_t(
                                             arg.a_m_k_(row, i).template unpack<>(ck::Number<0>{})))
                                      << " "
                                      << type_convert<float>(ck::f4_t(
                                             arg.a_m_k_(row, i).template unpack<>(ck::Number<1>{})))
                                      << " ";
                        }
                    }
                    else if constexpr(ck::is_same_v<ADataType, ck::f6x16_pk_t> ||
                                      ck::is_same_v<ADataType, ck::bf6x16_pk_t> ||
                                      ck::is_same_v<ADataType, ck::f6x32_pk_t> ||
                                      ck::is_same_v<ADataType, ck::bf6x32_pk_t>)
                    {
                        std::cout << type_convert<float>(arg.a_m_k_(row, i).unpack(
                                         i % ck::packed_size_v<ADataType>))
                                  << " ";
                    }
                    else
                    {
                        std::cout << type_convert<float>(arg.a_m_k_(row, i)) << " ";
                    }
                }
                std::cout << std::endl;
            }
#elif 0
            if(K < 600)
            {
                std::cout << std::endl;
                const auto row = 4;

                std::cout << "Ref GEMM a_m_k_scaled(" << row << ",:):";

                float sum = 0.0f;

                for(int i = 0; i < K; ++i)
                {
                    if(i % (std::max(16, ck::packed_size_v<ADataType>)) == 0)
                    {
                        std::cout << "\nRef GEMM a_m_k_scaled(" << row << ",:): sum = " << sum
                                  << std::endl;
                        std::cout << std::endl << "k = " << i << " : ";
                    }

                    std::cout << type_convert<float>(a_m_k_scaled(row, i)) << " ";

                    sum += type_convert<float>(a_m_k_scaled(row, i));
                }

                std::cout << std::endl;

                std::cout << "Ref GEMM a_m_k_scaled(" << row << ",:): sum = " << sum << std::endl;
            }
#endif
            for(int n = 0; n < N; n++)
            {
                for(int k = 0; k < K; k++)
                {
                    if constexpr(is_same_v<BDataType, f4x2_pk_t>)
                    {
                        // TODO: add support for RowMajor layout as well
                        if(k % 2 == 1)
                        {
                            continue;
                        }
                        auto b_pack = arg.b_k_n_(k, n);
                        auto b_scale =
                            type_convert<ComputeTypeB>(arg.b_kblock_n_scales_(k / SCALE_BLOCK, n));
                        auto b_f4_lo           = f4_t(b_pack.template unpack<>(Number<0>{}));
                        auto b_f4_hi           = f4_t(b_pack.template unpack<>(Number<1>{}));
                        b_k_n_scaled(k, n)     = type_convert<ComputeTypeB>(b_f4_lo) * b_scale;
                        b_k_n_scaled(k + 1, n) = type_convert<ComputeTypeB>(b_f4_hi) * b_scale;
                    }
                    else if constexpr(is_same_v<BDataType, f6x16_pk_t> ||
                                      is_same_v<BDataType, bf6x16_pk_t> ||
                                      is_same_v<BDataType, f6x32_pk_t> ||
                                      is_same_v<BDataType, bf6x32_pk_t>)
                    {
                        b_k_n_scaled(k, n) =
                            type_convert<ComputeTypeB>(
                                arg.b_k_n_(k, n).unpack(k % BDataType::packed_size)) *
                            type_convert<ComputeTypeB>(arg.b_kblock_n_scales_(k / SCALE_BLOCK, n));
                    }
                    else
                    {
                        b_k_n_scaled(k, n) =
                            type_convert<ComputeTypeB>(arg.b_k_n_(k, n)) *
                            type_convert<ComputeTypeB>(arg.b_kblock_n_scales_(k / SCALE_BLOCK, n));
                    }
                }
            }

            auto ref_gemm     = GemmInstance{};
            auto ref_invoker  = ref_gemm.MakeInvoker();
            auto ref_argument = ref_gemm.MakeArgument(a_m_k_scaled,
                                                      b_k_n_scaled,
                                                      arg.c_m_n_,
                                                      arg.a_element_op_,
                                                      arg.b_element_op_,
                                                      arg.c_element_op_);

            ref_invoker.Run(ref_argument);

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

    static auto MakeArgument(const Tensor<ADataType>& a_m_k,
                             const Tensor<ScaleDataType>& a_m_kblock_scales,
                             const Tensor<BDataType>& b_k_n,
                             const Tensor<ScaleDataType>& b_kblock_n_scales,
                             Tensor<CDataType>& c_m_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{a_m_k,
                        a_m_kblock_scales,
                        b_k_n,
                        b_kblock_n_scales,
                        c_m_n,
                        a_element_op,
                        b_element_op,
                        c_element_op};
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
        str << "ReferenceMXGemm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
