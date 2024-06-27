// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/library/tensor_operation_instance/add_device_operation_instance.hpp"
#include "ck/library/tensor_operation_instance/gpu/grouped_gemm_multiple_d/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_f16_f16_f16_mk_nk_mn_instance.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_tile_loop.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

using F16         = ck::half_t;
using Row         = ck::tensor_layout::gemm::RowMajor;
using Col         = ck::tensor_layout::gemm::ColumnMajor;
using Empty_Tuple = ck::Tuple<>;

using PassThrough                    = ck::tensor_operation::element_wise::PassThrough;
static constexpr auto GemmMNKPadding = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
static constexpr ck::index_t NumPrefetchK = 1;

void add_device_grouped_gemm_multi_d_splitk_cshuffle_f16_f16_f16_mk_nk_mn_mem_pv2_pf1_instances(
    std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
                                                  Col,
                                                  Empty_Tuple,
                                                  Row,
                                                  F16,
                                                  F16,
                                                  Empty_Tuple,
                                                  F16,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough>>>& instances)
{
    add_device_operation_instances(
        instances,
        device_ggemm_md_splitk_xdl_cshuffle_f16_f16_f16_mk_kn_mn_memory_instances<
            GemmMNKPadding,
            NumPrefetchK,
            ck::PipelineVersion::v2>{});
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
