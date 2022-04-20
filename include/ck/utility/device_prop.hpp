#pragma once

#include <string>

namespace ck {

std::string get_device_name()
{
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, device);
    const std::string name(props.gcnArchName);

    return name;
}

} // namespace ck
