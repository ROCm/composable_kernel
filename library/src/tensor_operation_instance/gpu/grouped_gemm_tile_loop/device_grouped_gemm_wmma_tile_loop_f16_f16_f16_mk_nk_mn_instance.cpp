// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_tile_loop/device_grouped_gemm_tile_loop_wmma_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using DsDataType = ck::Tuple<>;
using DsLayout   = ck::Tuple<>;

void add_device_grouped_gemm_wmma_tile_loop_f16_f16_f16_mk_nk_mn_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemmTileLoop<Row,
                                                          Col,
                                                          DsLayout,
                                                          Row,
                                                          F16,
                                                          F16,
                                                          DsDataType,
                                                          F16,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>>>& instances)
{

    add_device_grouped_gemm_tile_loop_wmma_instances<
        F16,
        Row,
        Col,
        device_grouped_gemm_tile_loop_wmma_mk_nk_mn_instances>(instances);
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
