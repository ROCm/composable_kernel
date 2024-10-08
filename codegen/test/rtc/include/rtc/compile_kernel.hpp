#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL

#include <ck/filesystem.hpp>
#include <rtc/kernel.hpp>
#include <functional>
#include <string>

namespace rtc {

struct src_file
{
    src_file(std::filesystem::path p, std::string c) : path{std::move(p)}, content{std::move(c)} {}
    CK::fs::path path;
    std::string content;
};

struct compile_options
{
    std::string flags       = "";
    std::string kernel_name = "main";
};

kernel compile_kernel(const std::vector<src_file>& srcs,
                      compile_options options = compile_options{});

} // namespace rtc

#endif
