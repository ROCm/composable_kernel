#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_COMPILE_KERNEL

#include <rtc/kernel.hpp>
#include <ck/filesystem.hpp>
#include <string>
#include <functional>

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

struct hip_compile_options
{
    std::size_t global;
    std::size_t local;
    std::string kernel_name                    = "kernel";
    std::string params                         = "";
    std::vector<src_file> additional_src_files = {};

    /**
     * @brief Set the launch parameters but allow v to override the values
     *
     * @param v A value class which can have a "global" and/or "local" keys to override the default
     * global and local
     * @param compute_global A function used to compute the global based on the local
     * @param default_local The defaul local to use if its missing from the v parameter
     */
    void set_launch_params(const std::function<std::size_t(std::size_t local)>& compute_global,
                           std::size_t default_local = 1024);

    void set_launch_params(std::size_t default_global, std::size_t default_local = 1024)
    {
        set_launch_params([=](auto) { return default_global; }, default_local);
    }
};

kernel compile_kernel(const std::vector<src_file>& src,
                      compile_options options = compile_options{});

kernel compile_hip_code_object(const std::string& content, hip_compile_options options);

} // namespace rtc

#endif
