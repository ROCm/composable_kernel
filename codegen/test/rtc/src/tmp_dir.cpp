#include <rtc/tmp_dir.hpp>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <thread>
#include <unistd.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>

namespace rtc {
std::string random_string(std::string::size_type length)
{
    static const std::string& chars = "0123456789"
                                      "abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    std::mt19937 rg{std::random_device{}()};
    std::uniform_int_distribution<std::string::size_type> pick(0, chars.length() - 1);

    std::string str(length, 0);
    std::generate(str.begin(), str.end(), [&] { return chars[pick(rg)]; });

    return str;
}

std::string unique_string(const std::string& prefix)
{
    auto pid = getpid();
    auto tid = std::this_thread::get_id();
    auto clk = std::chrono::steady_clock::now().time_since_epoch().count();
    std::stringstream ss;
    ss << std::hex << prefix << "-" << pid << "-" << tid << "-" << clk << "-" << random_string(16);
    return ss.str();
}

tmp_dir::tmp_dir(const std::string& prefix)
    : path(get_tmp_dir_path() + "/" + unique_string(prefix.empty() ? "ck-rtc" : "ck-rtc-" + prefix))
{
    new_dir(this->path);
}

std::string tmp_dir::get_tmp_dir_path()
{
    // use getenv to get the path of the tmp dir
    const char* tmp_dir_path = std::getenv("TMPDIR");

    if(tmp_dir_path == nullptr)
    {
        return "/tmp";
    }
    return tmp_dir_path;
}

// TODO: finish this method
void new_dir(std::string dir_path)
{
    int created = mkdir(dir_path.c_str(), 0755);
    if(created != 0)
    {
        throw std::runtime_error("Directory was not created");
    }
}

bool tmp_dir::exists(std::string check_path)
{
    struct stat sb;
    if(stat(check_path.c_str(), &sb) == 0 && !(sb.st_mode & S_IFDIR))
    {
        return true;
    }
    else
    {
        return false;
    }
}
void tmp_dir::execute(const std::string& cmd) const
{
    std::string s = "cd " + path + "; " + cmd;
    std::system(s.c_str());
}

// TODO: redo this method
tmp_dir::~tmp_dir()
{
    // std::filesystem::remove_all(this->path);
    ::remove(path.c_str());
}

} // namespace rtc
