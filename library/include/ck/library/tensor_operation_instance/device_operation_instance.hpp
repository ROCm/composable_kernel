#ifndef CK_DEVICE_OPERATION_INSTANCE_HPP
#define CK_DEVICE_OPERATION_INSTANCE_HPP

#include <stdlib.h>

namespace ck {
namespace tensor_operation {
namespace device {

template <typename OpInstance, typename NewOpInstances>
void add_device_operation_instances(std::vector<std::unique_ptr<OpInstance>>& op_instances,
                                    const NewOpInstances& new_op_instances)
{
    ck::static_for<0, std::tuple_size_v<NewOpInstances>, 1>{}([&](auto i) {
        const auto new_op_instance = std::get<i>(new_op_instances);

        using NewOpInstance = remove_cvref_t<decltype(new_op_instance)>;

        op_instances.push_back(std::make_unique<NewOpInstance>(new_op_instance));
    });
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
