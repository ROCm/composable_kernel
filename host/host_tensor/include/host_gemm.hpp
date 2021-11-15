#pragma once
#include "host_tensor.hpp"

template <typename AType, typename BType, typename CType>
void host_gemm_mk_kn_mn(const Tensor<AType>& a_m_k,
                        const Tensor<BType>& b_k_n,
                        Tensor<CType>& c_m_n)
{
    auto f_mk_kn_mn = [&](auto m, auto n) {
        const int K = a_m_k.mDesc.GetLengths()[1];

        double v = 0;

        for(int k = 0; k < K; ++k)
        {
            v += static_cast<const double>(a_m_k(m, k)) * static_cast<const double>(b_k_n(k, n));
        }

        c_m_n(m, n) = v;
    };

    make_ParallelTensorFunctor(f_mk_kn_mn,
                               c_m_n.mDesc.GetLengths()[0],
                               c_m_n.mDesc.GetLengths()[1])(std::thread::hardware_concurrency());
}
