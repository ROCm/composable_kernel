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
#include "ck/library/reference_tensor_operation/cpu/reference_groupnorm_bwd.hpp"

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
    ck::index_t N = 16;
    ck::index_t H = 16;
    ck::index_t W = 16;
    ck::index_t G = 32;
    ck::index_t C = 64;

    Tensor<DYDataType> dy({N, H, W, G, C});
    Tensor<XDataType> x({N, H, W, G, C});
    Tensor<GammaDataType> gamma({G, C});
    Tensor<MeanInvStdDataType> mean({N, G});
    Tensor<MeanInvStdDataType> inv_std({N, G});

    Tensor<DGammaDataType> dgamma({G, C});
    Tensor<DBetaDataType> dbeta({G, C});
    Tensor<DXDataType> dx({N, H, W, G, C});

    dy.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
    x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
    gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{0.0, 1.0});
    mean.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{0.0, 1.0});
    inv_std.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{0.0, 1.0});

    bool pass = true;
    {
        Tensor<DGammaDataType> host_dgamma({G, C});
        Tensor<DBetaDataType> host_dbeta({G, C});
        Tensor<DXDataType> host_dx({N, H, W, G, C});
        using ReferenceInstance = ck::tensor_operation::host::ReferenceGroupnorm<DYDataType,
                                                                                 XDataType,
                                                                                 GammaDataType,
                                                                                 MeanInvStdDataType,
                                                                                 DGammaDataType,
                                                                                 DBetaDataType,
                                                                                 DXDataType,
                                                                                 ComputeDataType>;

        ReferenceInstance ref;
        auto ref_argument = ref.MakeArgument(
            dy, x, gamma, mean, inv_std, host_dgamma, host_dbeta, host_dx, {N, H, W, G, C});
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);
    }

    return (pass ? 0 : 1);
}
