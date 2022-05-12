#ifndef REFERENCE_BATCHED_GEMM_HPP
#define REFERENCE_BATCHED_GEMM_HPP

#include <iostream>
#include <sstream>
#include "device_base.hpp"
#include "host_tensor.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
struct ReferenceBatchedGemm : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<ADataType>& a_g_m_k,
                 const Tensor<BDataType>& b_g_k_n,
                 Tensor<CDataType>& c_g_m_n,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op)
            : a_g_m_k_{a_g_m_k},
              b_g_k_n_{b_g_k_n},
              c_g_m_n_{c_g_m_n},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
        }

        const Tensor<ADataType>& a_g_m_k_;
        const Tensor<BDataType>& b_g_k_n_;
        Tensor<CDataType>& c_g_m_n_;

        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        using Argument = ReferenceBatchedGemm::Argument;

        float Run(const Argument& arg)
        {
            auto f_gmk_gkn_gmn = [&](auto g, auto m, auto n) {
                const int K = arg.a_g_m_k_.mDesc.GetLengths()[2];

                float v_acc = 0;

                for(int k = 0; k < K; ++k)
                {
                    float v_a;
                    float v_b;

                    arg.a_element_op_(v_a, static_cast<const float>(arg.a_g_m_k_(g, m, k)));
                    arg.b_element_op_(v_b, static_cast<const float>(arg.b_g_k_n_(g, k, n)));

                    v_acc += v_a * v_b;
                }

                float v_c;

                arg.c_element_op_(v_c, v_acc);

                arg.c_g_m_n_(g, m, n) = v_c;
            };

            make_ParallelTensorFunctor(f_gmk_gkn_gmn,
                                       arg.c_g_m_n_.mDesc.GetLengths()[0],
                                       arg.c_g_m_n_.mDesc.GetLengths()[1],
                                       arg.c_g_m_n_.mDesc.GetLengths()[2])(
                std::thread::hardware_concurrency());

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

    static auto MakeArgument(const Tensor<ADataType>& a_g_m_k,
                             const Tensor<BDataType>& b_g_k_n,
                             Tensor<CDataType>& c_g_m_n,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{a_g_m_k, b_g_k_n, c_g_m_n, a_element_op, b_element_op, c_element_op};
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
        str << "ReferenceBatchedGemm"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
#endif
