#ifndef GUARD_HOST_TEST_RTC_INCLUDE_RTC_TMP_DIR
#define GUARD_HOST_TEST_RTC_INCLUDE_RTC_TMP_DIR

#include <string>
#include <filesystem>

namespace rtc {

struct tmp_dir
{
    std::string path;
    tmp_dir(const std::string& prefix = "");

    std::string get_tmp_dir_path();
    void new_dir(std::string path);
    void execute(const std::string& cmd) const;

    tmp_dir(tmp_dir const&) = delete;
    tmp_dir& operator=(tmp_dir const&) = delete;

    ~tmp_dir();
};

} // namespace rtc

#endif
