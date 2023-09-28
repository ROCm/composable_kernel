#include <rtc/hip.hpp>
#include <stdexcept>

namespace rtc {

std::string hip_error(int error) { return hipGetErrorString(static_cast<hipError_t>(error)); }

int get_device_id()
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        throw std::runtime_error("No device");
    return device;
}

std::string get_device_name()
{
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, get_device_id());
    if(status != hipSuccess)
        throw std::runtime_error("Failed to get device properties");
    return props.gcnArchName;
}

} // namespace rtc
