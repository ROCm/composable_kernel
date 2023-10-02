// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_bwd_gamma_beta_impl.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_layernorm_bwd.hpp"

using DYDataType         = ck::half_t;
using XDataType          = ck::half_t;
using GammaDataType      = ck::half_t;
using MeanInvStdDataType = float;
using DGammaDataType     = ck::half_t;
using DBetaDataType      = ck::half_t;
using DXDataType         = ck::half_t;
using ComputeDataType    = float;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

int main()
{
    ck::index_t M = 1024;
    ck::index_t N = 1024;

    Tensor<DYDataType> dy({M, N});
    Tensor<XDataType> x({M, N});
    Tensor<GammaDataType> gamma({N});
    Tensor<MeanInvStdDataType> mean({M});
    Tensor<MeanInvStdDataType> inv_std({M});

    Tensor<DGammaDataType> dgamma({N});
    Tensor<DBetaDataType> dbeta({N});
    Tensor<DXDataType> dx({M, N});

    dy.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
    x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
    gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{0.0, 1.0});
    mean.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{0.0, 1.0});
    inv_std.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{0.0, 1.0});

    bool pass = true;
    {
        Tensor<DGammaDataType> host_dgamma({N});
        Tensor<DBetaDataType> host_dbeta({N});
        Tensor<DXDataType> host_dx({M, N});
        using ReferenceInstance =
            ck::tensor_operation::host::ReferenceLayernormBwd<DYDataType,
                                                              XDataType,
                                                              GammaDataType,
                                                              MeanInvStdDataType,
                                                              DGammaDataType,
                                                              DBetaDataType,
                                                              DXDataType,
                                                              ComputeDataType>;

        ReferenceInstance ref;
        auto ref_argument =
            ref.MakeArgument(dy, x, gamma, mean, inv_std, host_dgamma, host_dbeta, host_dx, {M, N});
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);
    }

    return (pass ? 0 : 1);
}
