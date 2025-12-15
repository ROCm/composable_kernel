// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <memory>
#include <type_traits>

#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#define CK_TILE_PIPELINE_COMPUTE_V3 1
#define CK_TILE_PIPELINE_MEMORY 2
#define CK_TILE_PIPELINE_COMPUTE_V4 3
#define CK_TILE_PIPELINE_COMPUTE_V5 4

namespace ck_tile {
namespace ops {

template <typename DeviceOp>
struct DeviceOperationInstanceFactory;

using NHWGC = ck_tile::tensor_layout::convolution::NHWGC;
using GKYXC = ck_tile::tensor_layout::convolution::GKYXC;
using NHWGK = ck_tile::tensor_layout::convolution::NHWGK;

using PassThrough = ck_tile::element_wise::PassThrough;

template <typename BaseOp, typename NewOpInstances>
void add_device_operation_instances(std::vector<std::unique_ptr<BaseOp>>& op_instances,
                                    const NewOpInstances& new_op_instances)
{
    ck_tile::static_for<0, std::tuple_size_v<NewOpInstances>, 1>{}([&](auto i) {
        const auto new_op_instance = std::get<i>(new_op_instances);

        using NewOpInstance = remove_cvref_t<decltype(new_op_instance)>;
        if constexpr(std::is_same_v<NewOpInstance, std::nullptr_t>)
        {
            return; // We can use nullptr_t to enable trailing comma
        }
        else
        {
            static_assert(std::is_base_of_v<BaseOp, NewOpInstance>,
                          "NewOpInstance must be derived from BaseOp");

            op_instances.push_back(std::make_unique<NewOpInstance>(new_op_instance));
        }
    });
}

} // namespace ops
} // namespace ck_tile
