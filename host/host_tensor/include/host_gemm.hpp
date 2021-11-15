#pragma once
#include "host_tensor.hpp"

template <>
void host_gemm<ushort, ushort, ushort>(const Tensor<ushort>& a,
                                       const Tensor<ushort>& b,
                                       Tensor<ushort>& c,
                                       const GemmMatrixLayout layout)
{
    if(layout == GemmMatrixLayout::MK_KN_MN)
    {
        auto f_mk_kn_mn = [&](auto m, auto n) {
            const int K = a.mDesc.GetLengths()[1];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += bfloat16_to_float(a(m, k)) * bfloat16_to_float(b(k, n));
            }

            c(m, n) = float_to_bfloat16(v);
        };

        make_ParallelTensorFunctor(f_mk_kn_mn, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::MK_NK_MN)
    {
        auto f_mk_nk_mn = [&](auto m, auto n) {
            const int K = a.mDesc.GetLengths()[1];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += bfloat16_to_float(a(m, k)) * bfloat16_to_float(b(n, k));
            }

            c(m, n) = float_to_bfloat16(v);
        };

        make_ParallelTensorFunctor(f_mk_nk_mn, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::KM_KN_MN)
    {
        auto f_km_kn_mn = [&](auto m, auto n) {
            const int K = a.mDesc.GetLengths()[0];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += bfloat16_to_float(a(k, m)) * bfloat16_to_float(b(k, n));
            }

            c(m, n) = float_to_bfloat16(v);
        };

        make_ParallelTensorFunctor(f_km_kn_mn, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::KM_NK_MN)
    {
        auto f_km_nk_mn = [&](auto m, auto n) {
            const int K = a.mDesc.GetLengths()[0];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += bfloat16_to_float(a(k, m)) * bfloat16_to_float(b(n, k));
            }

            c(m, n) = float_to_bfloat16(v);
        };

        make_ParallelTensorFunctor(f_km_nk_mn, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::MK_KN_NM)
    {
        auto f_mk_kn_nm = [&](auto n, auto m) {
            const int K = a.mDesc.GetLengths()[1];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += bfloat16_to_float(a(m, k)) * bfloat16_to_float(b(k, n));
            }

            c(n, m) = float_to_bfloat16(v);
        };

        make_ParallelTensorFunctor(f_mk_kn_nm, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::MK_NK_NM)
    {
        auto f_mk_nk_nm = [&](auto n, auto m) {
            const int K = a.mDesc.GetLengths()[1];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += bfloat16_to_float(a(m, k)) * bfloat16_to_float(b(n, k));
            }

            c(n, m) = float_to_bfloat16(v);
        };

        make_ParallelTensorFunctor(f_mk_nk_nm, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::KM_KN_NM)
    {
        auto f_km_kn_nm = [&](auto n, auto m) {
            const int K = a.mDesc.GetLengths()[0];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += bfloat16_to_float(a(k, m)) * bfloat16_to_float(b(k, n));
            }

            c(n, m) = float_to_bfloat16(v);
        };

        make_ParallelTensorFunctor(f_km_kn_nm, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else if(layout == GemmMatrixLayout::KM_NK_NM)
    {
        auto f_km_nk_nm = [&](auto n, auto m) {
            const int K = a.mDesc.GetLengths()[0];

            double v = 0;

            for(int k = 0; k < K; ++k)
            {
                v += bfloat16_to_float(a(k, m)) * bfloat16_to_float(b(n, k));
            }

            c(n, m) = float_to_bfloat16(v);
        };

        make_ParallelTensorFunctor(f_km_nk_nm, c.mDesc.GetLengths()[0], c.mDesc.GetLengths()[1])(
            std::thread::hardware_concurrency());
    }
    else
    {
        throw std::runtime_error("wrong! not supported layout");
    }
}

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
