#pragma once

#include <string>

namespace ck {

std::string get_device_name()
{
    hipDeviceProp_t props{};
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
    {
        return std::string();
    }

    hipGetDeviceProperties(&props, device);
    const std::string name(props.gcnArchName);

    return name;
}

} // namespace ck
