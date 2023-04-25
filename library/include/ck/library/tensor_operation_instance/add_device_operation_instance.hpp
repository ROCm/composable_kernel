// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <vector>
#include <type_traits>

#include "ck/utility/functional2.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

template <typename BaseOp, typename NewOpInstances>
void add_device_operation_instances(std::vector<std::unique_ptr<BaseOp>>& op_instances,
                                    const NewOpInstances& new_op_instances)
{
    ck::static_for<0, std::tuple_size_v<NewOpInstances>, 1>{}([&](auto i) {
        const auto new_op_instance = std::get<i>(new_op_instances);

        using NewOpInstance = remove_cvref_t<decltype(new_op_instance)>;

        static_assert(std::is_base_of_v<BaseOp, NewOpInstance>,
                      "wrong! NewOpInstance should be derived from BaseOp");

        op_instances.push_back(std::make_unique<NewOpInstance>(new_op_instance));
    });
}

enum struct ArchitectureEnum
{
    None,
    Gfx908,
    Gfx90a,
    Gfx1030,
    All
};
enum struct GemmFeatureEnum
{
    Xdl,
    Dl,
    Wmd
};
template <ArchitectureEnum... Is>
struct ArchitectureEnumSequence
{
    static constexpr int mSize = sizeof...(Is);

    __host__ __device__ static constexpr ArchitectureEnum At(int I)
    {
        // the last dummy element is to prevent compiler complain about empty array, when mSize = 0
        const ArchitectureEnum mData[mSize + 1] = {Is..., ArchitectureEnum::None};
        return mData[I];
    }
};
template <typename DeviceOp, GemmFeatureEnum Feature>
struct DeviceOperationInstances;

template <typename DeviceOp, typename Arch>
struct DeviceOperationInstanceCreator;
} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
