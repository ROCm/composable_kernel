#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_HIP
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_HIP

#include <hip/hip_runtime_api.h>
#include <string>

namespace rtc {

std::string get_device_name();
std::string hip_error(int error);

} // namespace rtc

#endif
