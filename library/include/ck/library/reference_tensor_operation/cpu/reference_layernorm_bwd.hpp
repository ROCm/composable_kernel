// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

namespace ck {
namespace tensor_operation {
namespace host {

template <typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename DGammaDataType,
          typename DBetaDataType,
          typename DXDataType,
          typename ComputeDataType>
struct ReferenceLayernorm : public device::BaseOperator
{
    // Argument
    struct Argument : public device::BaseArgument
    {
        Argument(const Tensor<DYDataType>& dy_m_k,
                 const Tensor<XDataType>& x_m_k,
                 const Tensor<GammaDataType>& gamma_k,
                 const Tensor<MeanInvStdDataType>& mean_m,
                 const Tensor<MeanInvStdDataType>& inv_std_m,
                 Tensor<DGammaDataType>& dgamma_k,
                 Tensor<DBetaDataType>& dbeta_k,
                 Tensor<DXDataType>& dx_m_k,
                 const std::vector<index_t> lengths)
            : dy_m_k_(dy_m_k),
              x_m_k_(x_m_k),
              gamma_k_(gamma_k),
              mean_m_(mean_m),
              inv_std_m_(inv_std_m),
              dgamma_k_(dgamma_k),
              dbeta_k_(dbeta_k),
              dx_m_k_(dx_m_k),
              lengths_(lengths)
        {
        }

        const Tensor<DYDataType>& dy_m_k_;
        const Tensor<XDataType>& x_m_k_;
        const Tensor<GammaDataType>& gamma_k_;
        const Tensor<MeanInvStdDataType>& mean_m_;
        const Tensor<MeanInvStdDataType>& inv_std_m_;
        Tensor<DGammaDataType>& dgamma_k_;
        Tensor<DBetaDataType>& dbeta_k_;
        Tensor<DXDataType>& dx_m_k_;
        std::vector<index_t> lengths_;
    };

    // Invoker
    struct Invoker : public device::BaseInvoker
    {
        float Run(const Argument& arg)
        {
            int M = arg.lengths_[0];
            int K = arg.lengths_[1];

            // Calculate dgamma and dbeta
            for(int k = 0; k < K; ++k)
            {
                ComputeDataType dgamma = 0;
                ComputeDataType dbeta  = 0;

                for(int m = 0; m < M; ++m)
                {
                    ComputeDataType dy   = ck::type_convert<ComputeDataType>(arg.dy_m_k_(m, k));
                    ComputeDataType x    = ck::type_convert<ComputeDataType>(arg.x_m_k_(m, k));
                    ComputeDataType mean = ck::type_convert<ComputeDataType>(arg.mean_m_(m));
                    ComputeDataType rstd = ck::type_convert<ComputeDataType>(arg.inv_std_m_(m));
                    dgamma += dy * rstd * (x - mean);
                    dbeta += dy;
                }
                arg.dgamma_k_(k) = ck::type_convert<DGammaDataType>(dgamma);
                arg.dbeta_k_(k)  = ck::type_convert<DBetaDataType>(dbeta);
            }

            // Calculate dx
            for(int m = 0; m < M; ++m)
            {
                ComputeDataType ds = 0;
                ComputeDataType db = 0;

                ComputeDataType mean = ck::type_convert<ComputeDataType>(arg.mean_m_(m));
                ComputeDataType rstd = ck::type_convert<ComputeDataType>(arg.inv_std_m_(m));

                for(int k = 0; k < K; ++k)
                {
                    ComputeDataType dy    = ck::type_convert<ComputeDataType>(arg.dy_m_k_(m, k));
                    ComputeDataType x     = ck::type_convert<ComputeDataType>(arg.x_m_k_(m, k));
                    ComputeDataType gamma = ck::type_convert<ComputeDataType>(arg.gamma_k_(k));

                    ds += dy * gamma * x;
                    db += dy * gamma;
                }

                for(int k = 0; k < K; ++k)
                {
                    ComputeDataType dy    = ck::type_convert<ComputeDataType>(arg.dy_m_k_(m, k));
                    ComputeDataType x     = ck::type_convert<ComputeDataType>(arg.x_m_k_(m, k));
                    ComputeDataType gamma = ck::type_convert<ComputeDataType>(arg.gamma_k_(k));

                    ComputeDataType b = (db * mean - ds) * rstd * rstd * rstd / K;
                    ComputeDataType c = -b * mean - db * rstd / K;

                    arg.dx_m_k_(m, k) = ck::type_convert<DXDataType>(dy * gamma * rstd + b * x + c);
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

    static auto MakeArgument(const Tensor<DYDataType>& dy_m_k,
                             const Tensor<XDataType>& x_m_k,
                             const Tensor<GammaDataType>& gamma_k,
                             const Tensor<MeanInvStdDataType>& mean_m,
                             const Tensor<MeanInvStdDataType>& inv_std_m,
                             Tensor<DGammaDataType>& dgamma_k,
                             Tensor<DBetaDataType>& dbeta_k,
                             Tensor<DXDataType>& dx_m_k,
                             const std::vector<index_t> lengths)
    {
        return Argument{
            dy_m_k, x_m_k, gamma_k, mean_m, inv_std_m, dgamma_k, dbeta_k, dx_m_k, lengths};
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
        str << "ReferenceLayernormBwd"
            << std::endl;
        // clang-format on

        return str.str();
    }
};

} // namespace host
} // namespace tensor_operation
} // namespace ck
