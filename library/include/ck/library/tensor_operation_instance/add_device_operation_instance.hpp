// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <type_traits>
#include <memory>

#include "ck/utility/functional2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename BaseOp, typename NewOpInstances>
void add_device_operation_instances(std::vector<std::unique_ptr<BaseOp>>& op_instances,
                                    const NewOpInstances& /*new_op_instances*/)
{
    ck::static_for<0, std::tuple_size_v<NewOpInstances>, 1>{}([&](auto i) {
        using NewOpInstance = std::tuple_element_t<i.value, NewOpInstances>;
        if constexpr(std::is_same_v<NewOpInstance, std::nullptr_t>)
        {
            return;
        }
        else
        {
            static_assert(std::is_base_of_v<BaseOp, NewOpInstance>,
                          "wrong! NewOpInstance should be derived from BaseOp");
            static_assert(std::is_default_constructible_v<NewOpInstance>,
                          "NewOpInstance must be default-constructible");
            op_instances.push_back(std::make_unique<NewOpInstance>());
        }
    });
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
