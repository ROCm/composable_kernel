// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>
#include <type_traits>
#include <memory>

#include "ck/utility/functional2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

/**
 * @brief Register device operation instances from a type container.
 * @tparam BaseOp The base class that all operation instances must derive from.
 * @tparam NewOpInstances A std::tuple (or ck::type_list) of device operation types.
 *         Only the type is used; the parameter value is unused (retained for type deduction).
 */
template <typename BaseOp, typename NewOpInstances>
void add_device_operation_instances(std::vector<std::unique_ptr<BaseOp>>& op_instances,
                                    const NewOpInstances& /*new_op_instances*/)
{
    ck::static_for<0, std::tuple_size_v<NewOpInstances>, 1>{}([&](auto i) {
        using NewOpInstance = std::tuple_element_t<i.value, NewOpInstances>;
        if constexpr(std::is_same_v<NewOpInstance, std::nullptr_t>)
        {
            return; // We can use nullptr_t to enable trailing comma
        }
        else
        {
            static_assert(std::is_base_of_v<BaseOp, NewOpInstance>,
                          "add_device_operation_instances: NewOpInstance must derive from BaseOp");
            static_assert(
                std::is_default_constructible_v<NewOpInstance>,
                "add_device_operation_instances: NewOpInstance must be default-constructible; "
                "registration default-constructs instances and ignores tuple values, so store "
                "configuration in template parameters instead of constructor arguments.");
            op_instances.push_back(std::make_unique<NewOpInstance>());
        }
    });
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
