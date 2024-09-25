#include "rtc/hip.hpp"
#include <rtc/compile_kernel.hpp>
#include <hip/hiprtc.h>
#include <rtc/tmp_dir.hpp>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <cassert>
#include <deque>
#include <numeric>

namespace rtc {

template <class T>
T generic_read_file(const std::string& filename, size_t offset = 0, size_t nbytes = 0)
{
    std::ifstream is(filename, std::ios::binary | std::ios::ate);
    if(nbytes == 0)
    {
        // if there is a non-zero offset and nbytes is not set,
        // calculate size of remaining bytes to read
        nbytes = is.tellg();
        if(offset > nbytes)
            throw std::runtime_error("offset is larger than file size");
        nbytes -= offset;
    }
    if(nbytes < 1)
        throw std::runtime_error("Invalid size for: " + filename);
    is.seekg(offset, std::ios::beg);

    T buffer(nbytes, 0);
    if(not is.read(&buffer[0], nbytes))
        throw std::runtime_error("Error reading file: " + filename);
    return buffer;
}

std::vector<char> read_buffer(const std::string& filename, size_t offset = 0, size_t nbytes = 0)
{
    return generic_read_file<std::vector<char>>(filename, offset, nbytes);
}

std::string read_string(const std::string& filename)
{
    return generic_read_file<std::string>(filename);
}

void write_buffer(const std::string& filename, const char* buffer, std::size_t size)
{
    std::ofstream os(filename);
    os.write(buffer, size);
}
void write_buffer(const std::string& filename, const std::vector<char>& buffer)
{
    write_buffer(filename, buffer.data(), buffer.size());
}
void write_string(const std::string& filename, const std::string_view& buffer)
{
    write_buffer(filename, buffer.data(), buffer.size());
}

std::string compiler() { return "/opt/rocm/llvm/bin/clang++ -x hip --cuda-device-only"; }
// TODO: undo after extracting the codeobj
// std::string compiler() { return "/opt/rocm/llvm/bin/clang++ -x hip"; }

kernel compile_kernel(const std::vector<src_file>& srcs, compile_options options)
{
    assert(not srcs.empty());
    tmp_dir td{"compile"};
    options.flags += " -I. -O3";
    options.flags += " -std=c++17";
    options.flags += " --offload-arch=" + get_device_name();
    std::string out;

    for(const auto& src : srcs)
    {
        CK::fs::path full_path   = td.path / src.path;
        CK::fs::path parent_path = full_path.parent_path();
        CK::fs::create_directories(parent_path);
        write_string(full_path.string(), src.content);
        if(src.path.extension().string() == ".cpp")
        {
            options.flags += " -c " + src.path.filename().string();
            if(out.empty())
                out = src.path.stem().string() + ".o";
        }
    }

    options.flags += " -o " + out;
    td.execute(compiler() + options.flags);

    auto out_path = td.path / out;
    if(not CK::fs::exists(out_path))
        throw std::runtime_error("Output file missing: " + out);

    auto obj = read_buffer(out_path.string());

    std::ofstream ofh("obj.o", std::ios::binary);
    for(auto i : obj)
        ofh << i;
    ofh.close();
    // int s = std::system(("/usr/bin/cp " + out_path.string() + " codeobj.bin").c_str());
    // assert(s == 0);
    return kernel{obj.data(), options.kernel_name};
}

struct hiprtc_src_file
{
    hiprtc_src_file() = default;
    hiprtc_src_file(const src_file& s) : path(s.path.string()), content(s.content) {}
    std::string path;
    std::string content;
    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.path, "path"), f(self.content, "content"));
    }
};

std::string hiprtc_error(hiprtcResult err, const std::string& msg)
{
    return "hiprtc: " + (hiprtcGetErrorString(err) + (": " + msg));
}

void hiprtc_check_error(hiprtcResult err, const std::string& msg, const std::string& ctx)
{
    if(err != HIPRTC_SUCCESS)
        throw std::runtime_error(hiprtc_error(err, msg));
}

// NOLINTNEXTLINE
#define MIGRAPHX_HIPRTC(...) \
    hiprtc_check_error(__VA_ARGS__, #__VA_ARGS__, "Lorem ipsum dolor sit amet")

#define MIGRAPHX_HIPRTC_THROW(error, msg) throw std::runtime_error(hiprtc_error(error, msg))

template <class F, F f> // NOLINT
struct manage_deleter
{
    template <class T>
    void operator()(T* x) const
    {
        if(x != nullptr)
        {
            (void)f(x);
        }
    }
};

template <class T, class F, F f> // NOLINT
using manage_ptr = std::unique_ptr<T, manage_deleter<F, f>>;

#define MIGRAPHX_MANAGE_PTR(T, F) manage_ptr<std::remove_pointer_t<T>, decltype(&F), &F> // NOLINT

// Workaround hiprtc's broken API
void hiprtc_program_destroy(hiprtcProgram prog) { hiprtcDestroyProgram(&prog); }
using hiprtc_program_ptr = MIGRAPHX_MANAGE_PTR(hiprtcProgram, hiprtc_program_destroy);

template <class... Ts>
hiprtc_program_ptr hiprtc_program_create(Ts... xs)
{
    hiprtcProgram prog = nullptr;
    auto result        = hiprtcCreateProgram(&prog, xs...);
    hiprtc_program_ptr p{prog};
    if(result != HIPRTC_SUCCESS)
        MIGRAPHX_HIPRTC_THROW(result, "Create program failed.");
    return p;
}

bool starts_with(const std::string& value, const std::string& prefix)
{
    if(prefix.size() > value.size())
        return false;
    else
        return std::equal(prefix.begin(), prefix.end(), value.begin());
}

bool ends_with(const std::string& value, const std::string& suffix)
{
    if(suffix.size() > value.size())
        return false;
    else
        return std::equal(suffix.rbegin(), suffix.rend(), value.rbegin());
}

std::vector<std::string> split_string(const std::string& s, char delim)
{
    std::vector<std::string> elems;
    std::stringstream ss(s + delim);
    std::string item;
    while(std::getline(ss, item, delim))
    {
        elems.push_back(item);
    }
    return elems;
}

template <class Strings>
inline std::string join_strings(Strings strings, const std::string& delim)
{
    auto it = strings.begin();
    if(it == strings.end())
        return "";

    auto nit = std::next(it);
    return std::accumulate(nit, strings.end(), *it, [&](std::string x, std::string y) {
        return std::move(x) + delim + std::move(y);
    });
}

struct hiprtc_program
{
    struct string_array
    {
        std::deque<std::string> strings{};
        std::vector<const char*> c_strs{};

        string_array() {}
        string_array(const string_array&) = delete;

        std::size_t size() const { return strings.size(); }

        const char** data() { return c_strs.data(); }

        void push_back(std::string s)
        {
            strings.push_back(std::move(s));
            c_strs.push_back(strings.back().c_str());
        }
    };

    hiprtc_program_ptr prog = nullptr;
    string_array headers{};
    string_array include_names{};
    std::string cpp_src  = "";
    std::string cpp_name = "";

    hiprtc_program(const std::string& src, const std::string& name = "main.cpp")
        : cpp_src(src), cpp_name(name)
    {
        create_program();
    }

    hiprtc_program(std::vector<src_file> srcs)
    {
        for(auto&& src : srcs)
        {
            if(ends_with(src.path, ".cpp"))
            {
                cpp_src  = std::move(src.content);
                cpp_name = std::move(src.path);
            }
            else
            {
                headers.push_back(std::move(src.content));
                include_names.push_back(std::move(src.path));
            }
        }
        create_program();
    }

    void create_program()
    {
        assert(not cpp_src.empty());
        assert(not cpp_name.empty());
        assert(headers.size() == include_names.size());
        prog = hiprtc_program_create(cpp_src.c_str(),
                                     cpp_name.c_str(),
                                     headers.size(),
                                     headers.data(),
                                     include_names.data());
    }

    void compile(const std::vector<std::string>& options, bool quiet = false) const
    {
        // if(enabled(MIGRAPHX_TRACE_HIPRTC{}))
        //     std::cout << "hiprtc " << join_strings(options, " ") << " " << cpp_name << std::endl;
        std::vector<const char*> c_options;
        std::transform(options.begin(),
                       options.end(),
                       std::back_inserter(c_options),
                       [](const std::string& s) { return s.c_str(); });
        std::cout << "BEFORE HIPRTC COMPILE" << std::endl;
        auto result   = hiprtcCompileProgram(prog.get(), c_options.size(), c_options.data());
        auto prog_log = log();
        if(not prog_log.empty() and not quiet)
        {
            std::cerr << prog_log << std::endl;
        }
        if(result != HIPRTC_SUCCESS)
            throw std::runtime_error("Compilation failed.");
    }

    std::string log() const
    {
        std::size_t n = 0;
        MIGRAPHX_HIPRTC(hiprtcGetProgramLogSize(prog.get(), &n));
        if(n == 0)
            return {};
        std::string buffer(n, '\0');
        MIGRAPHX_HIPRTC(hiprtcGetProgramLog(prog.get(), buffer.data()));
        assert(buffer.back() != 0);
        return buffer;
    }

    std::vector<char> get_code_obj() const
    {
        std::size_t n = 0;
        MIGRAPHX_HIPRTC(hiprtcGetCodeSize(prog.get(), &n));
        std::vector<char> buffer(n);
        MIGRAPHX_HIPRTC(hiprtcGetCode(prog.get(), buffer.data()));
        return buffer;
    }
};

std::vector<std::vector<char>> compile_hip_src_with_hiprtc(std::vector<src_file> srcs,
                                                           const std::string& params,
                                                           const std::string& arch)
{
    hiprtc_program prog(std::move(srcs));
    auto options = split_string(params, ' ');
    options.push_back("-DMIGRAPHX_USE_HIPRTC=1");
    if(true)
    {
        options.push_back("-DMIGRAPHX_HAS_DPP=0");
        options.push_back("-DMIGRAPHX_ENABLE_HIPRTC_WORKAROUNDS=1");
        options.push_back("-Wno-reserved-identifier");
        options.push_back("-Wno-unused-parameter");
        options.push_back("-Wno-gnu-line-marker");
        options.push_back("-Wno-old-style-cast");
    }
    if(true)
        options.push_back("-DMIGRAPHX_DEBUG");
    if(std::none_of(options.begin(), options.end(), [](const std::string& s) {
           return starts_with(s, "--std=") or starts_with(s, "-std=");
       }))
        options.push_back("-std=c++17");
    options.push_back("-fno-gpu-rdc");
    options.push_back("-O3");
    options.push_back("-Wno-cuda-compat");
    options.push_back("--offload-arch=" + arch);
    prog.compile(options);
    return {prog.get_code_obj()};
}

bool hip_has_flags(const std::vector<std::string>& flags)
{
    hiprtc_program prog{" "};
    try
    {
        prog.compile(flags, true);
        return true;
    }
    catch(...)
    {
        return false;
    }
}

bool hip_accept_non_uniform_wg()
{
    static bool non_uniform_wg = hip_has_flags({"-fno-offload-uniform-block"});
    return non_uniform_wg;
}

static std::vector<std::string> get_compiler_warnings()
{
    std::vector<std::string> warnings = {
        "-Weverything",
        "-Wno-c++98-compat",
        "-Wno-c++98-compat-pedantic",
        "-Wno-conversion",
        "-Wno-double-promotion",
        "-Wno-exit-time-destructors",
        "-Wno-extra-semi",
        "-Wno-extra-semi-stmt",
        "-Wno-float-conversion",
        "-Wno-gnu-anonymous-struct",
        "-Wno-gnu-zero-variadic-macro-arguments",
        "-Wno-missing-prototypes",
        "-Wno-nested-anon-types",
        "-Wno-padded",
        "-Wno-shorten-64-to-32",
        "-Wno-sign-conversion",
        "-Wno-sign-compare",
        "-Wno-unused-command-line-argument",
        "-Wno-weak-vtables",
        "-Wno-c99-extensions",
    };

    if(hip_has_flags({"-Werror", "-Wunsafe-buffer-usage"}))
        warnings.push_back("-Wno-unsafe-buffer-usage");
    return warnings;
}

const std::vector<std::string>& compiler_warnings()
{
    static std::vector<std::string> warnings = get_compiler_warnings();
    return warnings;
}

kernel compile_hip_code_object(const std::string& content, hip_compile_options options)
{
    assert(options.global > 0);
    assert(options.local > 0);
    // assert(not options.inputs.empty());
    // assert(options.inputs.size() == options.virtual_inputs.size() or
    //        options.virtual_inputs.empty());
    std::vector<src_file> srcs = options.additional_src_files;
    // Neko sranje
    // static auto kernels{::migraphx_kernels()};
    // std::transform(
    //     kernels.begin(),
    //     kernels.end(),
    //     std::back_inserter(srcs),
    //     [](const std::pair<std::string_view, std::string_view>& elem) { return src_file{elem};
    //     });
    srcs.emplace_back("main.cpp", content);

    for (auto src : srcs) {
        std::cout << src.path << std::endl;
    }


    // auto args_hpp =
    //     generate_args_hpp(options.virtual_inputs.empty() ? options.inputs :
    //     options.virtual_inputs);
    // srcs.emplace_back("args.hpp", args_hpp);

    if(options.global % options.local != 0 and hip_accept_non_uniform_wg())
        options.params += " -fno-offload-uniform-block";
    else
        assert(options.global % options.local == 0);

    options.params += " -DMIGRAPHX_NGLOBAL=" + std::to_string(options.global);
    options.params += " -DMIGRAPHX_NLOCAL=" + std::to_string(options.local);
    options.params += " " + join_strings(compiler_warnings(), " ");
    options.params += " -ftemplate-backtrace-limit=0";
    options.params += " -Werror";
    auto cos = compile_hip_src_with_hiprtc(srcs, options.params, get_device_name());
    if(cos.size() != 1)
        std::runtime_error("No code object");
    auto& obj = cos.front(); 

    return kernel{obj.data(), options.kernel_name};
}

} // namespace rtc
