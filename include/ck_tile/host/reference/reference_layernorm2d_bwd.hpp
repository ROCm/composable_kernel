#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include <thread>

namespace ck_tile {

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename ComputeDataType,
          typename YDataType,
          typename MeanDataType,
          typename InvStdDataType>
CK_TILE_HOST void reference_layernorm2d_bwd_gamma_part(const HostTensor<XDataType>& x_m_n,
                                                       const HostTensor<YDataType>& dy_m_n,
                                                       const HostTensor<GammaDataType>& gamma_n,
                                                       const HostTensor<MeanDataType>& mean_m,
                                                       const HostTensor<InvStdDataType>& inv_std_m,
                                                       HostTensor<GammaDataType>& dgamma_mpart_n,
                                                       HostTensor<BetaDataType>& dbeta_mpart_n,
                                                       HostTensor<XDataType>& dx_m_n,
                                                       
                                                       //tmp
                                                       HostTensor<ComputeDataType>& ds_m,
                                                       HostTensor<ComputeDataType>& db_m)
{
    
    const auto MN = x_m_n.mDesc.get_lengths();
    const int M = MN[0];
    const int N = MN[1];
    const int PartM = dgamma_mpart_n.mDesc.get_lengths()[0];
    const int MLoop = (M + PartM - 1) / PartM;
    printf("\ndteng print---M=%d,N=%d,PartM=%d,MLoop=%d\n",M,N,PartM,MLoop);
    auto f = [&](auto m) {
        const int m_offset = m * MLoop;
        //calculate dgamma, dbeta
        for(int n = 0; n < N; ++n)
        {
            ComputeDataType gamma_acc = 0;
            ComputeDataType beta_acc = 0;
            for(int inner_m = 0; inner_m < MLoop && m_offset + inner_m < M; inner_m++) 
            {
                const ComputeDataType mean = ck_tile::type_convert<ComputeDataType>(mean_m(m_offset + inner_m));
                const ComputeDataType inv_std = ck_tile::type_convert<ComputeDataType>(inv_std_m(m_offset + inner_m));
                const ComputeDataType x = ck_tile::type_convert<ComputeDataType>(x_m_n(m_offset + inner_m, n));
                const ComputeDataType dy = ck_tile::type_convert<ComputeDataType>(dy_m_n(m_offset + inner_m, n));
                gamma_acc += dy * (x - mean) * inv_std;
                beta_acc += dy;
            }

            dgamma_mpart_n(m, n) = ck_tile::type_convert<GammaDataType>(gamma_acc);
            dbeta_mpart_n(m, n) = ck_tile::type_convert<BetaDataType>(beta_acc);
        }

        //calculate dx
        for(int inner_m = 0; inner_m < MLoop && m_offset + inner_m < M; inner_m++)
        {
            ComputeDataType ds = 0;
            ComputeDataType db = 0;
            const ComputeDataType mean = ck_tile::type_convert<ComputeDataType>(mean_m(m_offset + inner_m));
            const ComputeDataType inv_std = ck_tile::type_convert<ComputeDataType>(inv_std_m(m_offset + inner_m));
            for(int n = 0; n < N; ++n)
            {
                const ComputeDataType dy    = ck_tile::type_convert<ComputeDataType>(dy_m_n(m_offset + inner_m, n));
                const ComputeDataType x     = ck_tile::type_convert<ComputeDataType>(x_m_n(m_offset + inner_m, n));
                const ComputeDataType gamma = ck_tile::type_convert<ComputeDataType>(gamma_n(n));
                ds += dy * gamma * x;
                db += dy * gamma;
            }
            ComputeDataType b = (db * mean - ds) * inv_std * inv_std * inv_std / N;
            ComputeDataType c = -b * mean - db * inv_std / N;
            for(int n = 0; n < N; ++n)
            {
                const ComputeDataType dy    = ck_tile::type_convert<ComputeDataType>(dy_m_n(m_offset + inner_m, n));
                const ComputeDataType x     = ck_tile::type_convert<ComputeDataType>(x_m_n(m_offset + inner_m, n));
                const ComputeDataType gamma = ck_tile::type_convert<ComputeDataType>(gamma_n(n));
                dx_m_n(m_offset + inner_m, n) = ck_tile::type_convert<XDataType>(dy * gamma * inv_std + b * x + c);
            }
        }
    };

    make_ParallelTensorFunctor(f, PartM)(std::thread::hardware_concurrency());
}
} // namespace ck_tile
