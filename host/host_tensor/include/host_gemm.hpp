#pragma once
#include "host_tensor.hpp"

template <typename AType,
          typename BType,
          typename CType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation>
void host_gemm_mk_kn_mn(const Tensor<AType>& a_m_k,
                        const Tensor<BType>& b_k_n,
                        Tensor<CType>& c_m_n,
                        const AElementwiseOperation& a_element_op,
                        const BElementwiseOperation& b_element_op,
                        const CElementwiseOperation& c_element_op)
{
    auto f_mk_kn_mn = [&](auto m, auto n) {
        const int K = a_m_k.mDesc.GetLengths()[1];

        double v = 0;

        for(int k = 0; k < K; ++k)
        {
            v += static_cast<const double>(a_element_op(a_m_k(m, k))) *
                 static_cast<const double>(b_element_op(b_k_n(k, n)));
        }

        c_m_n(m, n) = c_element_op(v);
    };

    make_ParallelTensorFunctor(f_mk_kn_mn,
                               c_m_n.mDesc.GetLengths()[0],
                               c_m_n.mDesc.GetLengths()[1])(std::thread::hardware_concurrency());
}
