#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL

#include <rtc/kernel.hpp>
#include <ck/filesystem.hpp>
#include <string>
#include <functional>

namespace rtc {

struct src_file
{
    CK::fs::path path;
    std::string content;
};

struct compile_options
{
    std::string flags       = "";
    std::string kernel_name = "main";
    std::vector<src_file> additional_src_files = {};
    std::string params = "";
};

kernel compile_kernel(const std::vector<src_file>& src,
                      compile_options options = compile_options{});

kernel compile_kernel(const std::string& content, compile_options options = compile_options{});
                    
} // namespace rtc

#endif
