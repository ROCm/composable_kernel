// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <sstream>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wno-unknown-warning-option"
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck {
namespace tensor_operation {
namespace host {

template <ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ComputeDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          ck::enable_if_t<(NumDimM == 2 || NumDimM == 6) && (NumDimN == 2 || NumDimN == 6) &&
                              (NumDimK == 2 || NumDimK == 6),
                          bool> = false>
struct ReferenceContraction_M2_N2_K2 : public ck::tensor_operation::device::BaseOperator
{
    // Argument
    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_ms_ks,
                 const Tensor<BDataType>& b_ns_ks,
                 Tensor<CDataType>& c_ms_ns,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op)
            : a_ms_ks_{a_ms_ks},
              b_ns_ks_{b_ns_ks},
              c_ms_ns_{c_ms_ns},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op}
        {
        }

        const Tensor<ADataType>& a_ms_ks_;
        const Tensor<BDataType>& b_ns_ks_;
        Tensor<CDataType>& c_ms_ns_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
    };

    // Invoker
    struct Invoker : public ck::tensor_operation::device::BaseInvoker
    {
        using Argument = ReferenceContraction_M2_N2_K2::Argument;

        float Run(const Argument& arg)
        {
            auto f_ms_ns = [&](auto m0,
                               auto m1,
                               auto m2,
                               auto m3,
                               auto m4,
                               auto m5,
                               auto n0,
                               auto n1,
                               auto n2,
                               auto n3,
                               auto n4,
                               auto n5) {
                const ck::index_t K0 = arg.a_ms_ks_.mDesc.GetLengths()[NumDimM];
                const ck::index_t K1 = arg.a_ms_ks_.mDesc.GetLengths()[NumDimM + 1];
                const ck::index_t K2 =
                    NumDimK >= 3 ? arg.a_ms_ks_.mDesc.GetLengths()[NumDimM + 2] : 1;
                const ck::index_t K3 =
                    NumDimK >= 4 ? arg.a_ms_ks_.mDesc.GetLengths()[NumDimM + 3] : 1;
                const ck::index_t K4 =
                    NumDimK >= 5 ? arg.a_ms_ks_.mDesc.GetLengths()[NumDimM + 4] : 1;
                const ck::index_t K5 =
                    NumDimK >= 6 ? arg.a_ms_ks_.mDesc.GetLengths()[NumDimM + 5] : 1;

                AccDataType v_acc = 0;

                for(ck::index_t k0 = 0; k0 < K0; ++k0)
                {
                    for(ck::index_t k1 = 0; k1 < K1; ++k1)
                    {
                        for(ck::index_t k2 = 0; k2 < K2; ++k2)
                        {
                            for(ck::index_t k3 = 0; k3 < K3; ++k3)
                            {
                                for(ck::index_t k4 = 0; k4 < K4; ++k4)
                                {
                                    for(ck::index_t k5 = 0; k5 < K5; ++k5)
                                    {
                                        ComputeDataType v_a_compute_input;
                                        ComputeDataType v_b_compute_input;

                                        // Simulate the possible casting when ComputeDataType is
                                        // different than the A/B data types
                                        if constexpr(NumDimK == 2)
                                        {
                                            v_a_compute_input = ck::type_convert<ComputeDataType>(
                                                arg.a_ms_ks_(m0, m1, k0, k1));
                                            v_b_compute_input = ck::type_convert<ComputeDataType>(
                                                arg.b_ns_ks_(n0, n1, k0, k1));
                                        }
                                        else if constexpr(NumDimK == 6)
                                        {
                                            v_a_compute_input = ck::type_convert<
                                                ComputeDataType>(arg.a_ms_ks_(
                                                m0, m1, m2, m3, m4, m5, k0, k1, k2, k3, k4, k5));
                                            v_b_compute_input = ck::type_convert<
                                                ComputeDataType>(arg.b_ns_ks_(
                                                n0, n1, n2, n3, n4, n5, k0, k1, k2, k3, k4, k5));
                                        }

                                        AccDataType v_a;
                                        AccDataType v_b;

                                        arg.a_element_op_(
                                            v_a, ck::type_convert<AccDataType>(v_a_compute_input));
                                        arg.b_element_op_(
                                            v_b, ck::type_convert<AccDataType>(v_b_compute_input));

                                        v_acc += v_a * v_b;
                                    }
                                }
                            }
                        }
                    }
                }

                if constexpr(NumDimK == 2)
                {
                    arg.c_ms_ns_(m0, m1, n0, n1) = ck::type_convert<CDataType>(v_acc);
                }
                else if constexpr(NumDimK == 6)
                {
                    arg.c_ms_ns_(m0, m1, m2, m3, m4, m5, n0, n1, n2, n3, n4, n5) =
                        ck::type_convert<CDataType>(v_acc);
                }
            };

            if constexpr(NumDimK == 2)
            {
                make_ParallelTensorFunctor(f_ms_ns,
                                           arg.c_ms_ns_.mDesc.GetLengths()[0],
                                           arg.c_ms_ns_.mDesc.GetLengths()[1],
                                           1,
                                           1,
                                           1,
                                           1,
                                           arg.c_ms_ns_.mDesc.GetLengths()[2],
                                           arg.c_ms_ns_.mDesc.GetLengths()[3],
                                           1,
                                           1,
                                           1,
                                           1)(std::thread::hardware_concurrency());
            }
            else if constexpr(NumDimK == 6)
            {
                make_ParallelTensorFunctor(f_ms_ns,
                                           arg.c_ms_ns_.mDesc.GetLengths()[0],
                                           arg.c_ms_ns_.mDesc.GetLengths()[1],
                                           arg.c_ms_ns_.mDesc.GetLengths()[2],
                                           arg.c_ms_ns_.mDesc.GetLengths()[3],
                                           arg.c_ms_ns_.mDesc.GetLengths()[4],
                                           arg.c_ms_ns_.mDesc.GetLengths()[5],
                                           arg.c_ms_ns_.mDesc.GetLengths()[6],
                                           arg.c_ms_ns_.mDesc.GetLengths()[7],
                                           arg.c_ms_ns_.mDesc.GetLengths()[8],
                                           arg.c_ms_ns_.mDesc.GetLengths()[9],
                                           arg.c_ms_ns_.mDesc.GetLengths()[10],
                                           arg.c_ms_ns_.mDesc.GetLengths()[11])(
                    std::thread::hardware_concurrency());
            }

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
                             Tensor<CDataType>& c_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op)
    {
        return Argument{a_ms_ks, b_ns_ks, c_ms_ns, a_element_op, b_element_op};
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

// hardcoded for NumDimG == 1, NumDimM == 2, NumDimN == 3, NumDimK == 1
template <ck::index_t NumDimG,
          ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ck::enable_if_t<NumDimG == 1 && NumDimM == 2 && NumDimN == 3 && NumDimK == 1, bool> =
              false>
struct ReferenceBatchedContraction_G1_M2_N3_K1 : public ck::tensor_operation::device::BaseOperator
{
    // Argument
    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_gs_ms_ks,
                 const Tensor<BDataType>& b_gs_ns_ks,
                 Tensor<EDataType>& e_gs_ms_ns,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : a_gs_ms_ks_{a_gs_ms_ks},
              b_gs_ns_ks_{b_gs_ns_ks},
              e_gs_ms_ns_{e_gs_ms_ns},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
        }

        const Tensor<ADataType>& a_gs_ms_ks_;
        const Tensor<BDataType>& b_gs_ns_ks_;
        Tensor<EDataType>& e_gs_ms_ns_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public ck::tensor_operation::device::BaseInvoker
    {
        using Argument = ReferenceBatchedContraction_G1_M2_N3_K1::Argument;

        float Run(const Argument& arg)
        {
            auto f_gs_ms_ns = [&](auto g0, auto m0, auto m1, auto n0, auto n1, auto n2) {
                const int K0 = arg.a_gs_ms_ks_.mDesc.GetLengths()[3];

                AccDataType v_acc = 0;

                for(int k0 = 0; k0 < K0; ++k0)
                {
                    AccDataType v_a;
                    AccDataType v_b;

                    arg.a_element_op_(
                        v_a, ck::type_convert<const AccDataType>(arg.a_gs_ms_ks_(g0, m0, m1, k0)));
                    arg.b_element_op_(
                        v_b,
                        ck::type_convert<const AccDataType>(arg.b_gs_ns_ks_(g0, n0, n1, n2, k0)));

                    v_acc += v_a * v_b;
                }

                AccDataType v_c;

                arg.cde_element_op_(v_c, v_acc);

                arg.e_gs_ms_ns_(g0, m0, m1, n0, n1, n2) = v_c;
            };

            make_ParallelTensorFunctor(f_gs_ms_ns,
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[0],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[1],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[2],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[3],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[4],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[5])(
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

    static auto MakeArgument(const Tensor<ADataType>& a_gs_ms_ks,
                             const Tensor<BDataType>& b_gs_ns_ks,
                             Tensor<EDataType>& e_gs_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{
            a_gs_ms_ks, b_gs_ns_ks, e_gs_ms_ns, a_element_op, b_element_op, cde_element_op};
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
        str << "ReferenceBatchedContraction_G1_M3_N2_K1"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

template <ck::index_t NumDimG,
          ck::index_t NumDimM,
          ck::index_t NumDimN,
          ck::index_t NumDimK,
          typename ADataType,
          typename BDataType,
          typename EDataType,
          typename AccDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ck::enable_if_t<NumDimG == 1 && NumDimM == 3 && NumDimN == 2 && NumDimK == 1, bool> =
              false>
struct ReferenceBatchedContraction_G1_M3_N2_K1 : public ck::tensor_operation::device::BaseOperator
{
    // Argument
    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_gs_ms_ks,
                 const Tensor<BDataType>& b_gs_ns_ks,
                 Tensor<EDataType>& e_gs_ms_ns,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CDEElementwiseOperation cde_element_op)
            : a_gs_ms_ks_{a_gs_ms_ks},
              b_gs_ns_ks_{b_gs_ns_ks},
              e_gs_ms_ns_{e_gs_ms_ns},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op}
        {
        }

        const Tensor<ADataType>& a_gs_ms_ks_;
        const Tensor<BDataType>& b_gs_ns_ks_;
        Tensor<EDataType>& e_gs_ms_ns_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;
    };

    // Invoker
    struct Invoker : public ck::tensor_operation::device::BaseInvoker
    {
        using Argument = ReferenceBatchedContraction_G1_M3_N2_K1::Argument;

        float Run(const Argument& arg)
        {
            auto f_gs_ms_ns = [&](auto g0, auto m0, auto m1, auto m2, auto n0, auto n1) {
                const int K0 = arg.a_gs_ms_ks_.mDesc.GetLengths()[4];

                AccDataType v_acc = 0;

                for(int k0 = 0; k0 < K0; ++k0)
                {
                    AccDataType v_a;
                    AccDataType v_b;

                    arg.a_element_op_(
                        v_a,
                        ck::type_convert<const AccDataType>(arg.a_gs_ms_ks_(g0, m0, m1, m2, k0)));
                    arg.b_element_op_(
                        v_b, ck::type_convert<const AccDataType>(arg.b_gs_ns_ks_(g0, n0, n1, k0)));

                    v_acc += v_a * v_b;
                }

                AccDataType v_c;

                arg.cde_element_op_(v_c, v_acc);

                arg.e_gs_ms_ns_(g0, m0, m1, m2, n0, n1) = v_c;
            };

            make_ParallelTensorFunctor(f_gs_ms_ns,
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[0],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[1],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[2],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[3],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[4],
                                       arg.e_gs_ms_ns_.mDesc.GetLengths()[5])(
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

    static auto MakeArgument(const Tensor<ADataType>& a_gs_ms_ks,
                             const Tensor<BDataType>& b_gs_ns_ks,
                             Tensor<EDataType>& e_gs_ms_ns,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CDEElementwiseOperation cde_element_op)
    {
        return Argument{
            a_gs_ms_ks, b_gs_ns_ks, e_gs_ms_ns, a_element_op, b_element_op, cde_element_op};
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
        str << "ReferenceBatchedContraction_G1_M3_N2_K1"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck

#pragma clang diagnostic pop
