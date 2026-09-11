// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "profile_grouped_gemm_tile_loop_generic_impl.hpp"

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename DDataType,
          typename EDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename DLayout,
          typename ELayout>
bool profile_grouped_gemm_multiply_tile_loop_impl(int do_verification,
                                                  int init_method,
                                                  bool do_log,
                                                  bool time_kernel,
                                                  const std::vector<int>& Ms,
                                                  const std::vector<int>& Ns,
                                                  const std::vector<int>& Ks,
                                                  const std::vector<int>& StrideAs,
                                                  const std::vector<int>& StrideBs,
                                                  const std::vector<int>& StrideDs,
                                                  const std::vector<int>& StrideEs,
                                                  int n_warmup = 10,
                                                  int n_iter   = 50)
{
    std::vector<std::array<int, 1>> stride_ds;
    for(size_t i = 0; i < StrideDs.size(); ++i)
    {
        stride_ds.emplace_back(std::array<int, 1>{StrideDs[i]});
    }

    return profile_grouped_gemm_tile_loop_generic_impl<
        ADataType,
        BDataType,
        Tuple<DDataType>,
        EDataType,
        ALayout,
        BLayout,
        Tuple<DLayout>,
        ELayout,
        PassThrough,
        PassThrough,
        ck::tensor_operation::element_wise::Multiply>(do_verification,
                                                      init_method,
                                                      do_log,
                                                      time_kernel,
                                                      Ms,
                                                      Ns,
                                                      Ks,
                                                      StrideAs,
                                                      StrideBs,
                                                      stride_ds,
                                                      StrideEs,
                                                      n_warmup,
                                                      n_iter);
}

} // namespace profiler
} // namespace ck
