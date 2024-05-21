#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_op.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/conv_fwd_problem.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/tensor_operation/gpu/device/helper.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <random>
#include <test.hpp>
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <fstream>

std::vector<rtc::src_file> get_headers_for_test()
{
    std::vector<rtc::src_file> result;
    auto hs = ck::host::GetHeaders();
    std::transform(
        hs.begin(), hs.end(), std::back_inserter(result), [&](const auto& p) -> rtc::src_file {
            return {p.first, p.second};
        });
    return result;
}

template <typename V>
std::size_t GetSize(V mLens, V mStrides)
{
    std::size_t space = 1;
    for(std::size_t i = 0; i < mLens.Size(); ++i)
    {
        if(mLens[i] == 0)
            continue;

        space += (mLens[i] - 1) * mStrides[i];
    }
    return space;
}

template <class T, typename V>
rtc::buffer<T> generate_buffer(std::size_t n, V mLens, V mStrides, std::size_t seed = 0)
{
    std::size_t space = GetSize(mLens, mStrides);
    rtc::buffer<T> result(space);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-1.0);
    std::generate(result.begin(), result.end(), [&] { return dis(gen); });
    // std::fill(result.begin(), result.end(), 1);
    return result;
}

template <class T, class U>
bool allclose(const T& a, const U& b, double atol = 0.01, double rtol = 0.01)
{
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [&](double x, double y) {
        return fabs(x - y) < atol + rtol * fabs(y);
    });
}

std::string classify(double x)
{
    switch(std::fpclassify(x))
    {
    case FP_INFINITE: return "inf";
    case FP_NAN: return "nan";
    case FP_NORMAL: return "normal";
    case FP_SUBNORMAL: return "subnormal";
    case FP_ZERO: return "zero";
    default: return "unknown";
    }
}

template <class Buffer>
void print_classification(const Buffer& x)
{
    std::unordered_set<std::string> result;
    for(const auto& i : x)
        result.insert(classify(i));
    for(const auto& c : result)
        std::cout << c << ", ";
    std::cout << std::endl;
}

template <class Buffer>
void print_statistics(const Buffer& x)
{
    std::cout << "Min value: " << *std::min_element(x.begin(), x.end()) << ", ";
    std::cout << "Max value: " << *std::max_element(x.begin(), x.end()) << ", ";
    double num_elements = x.size();
    auto mean =
        std::accumulate(x.begin(), x.end(), double{0.0}, std::plus<double>{}) / num_elements;
    auto stddev = std::sqrt(
        std::accumulate(x.begin(),
                        x.end(),
                        double{0.0},
                        [&](double r, double v) { return r + std::pow((v - mean), 2.0); }) /
        num_elements);
    std::cout << "Mean: " << mean << ", ";
    std::cout << "StdDev: " << stddev << "\n";
}

template <class Buffer>
void print_preview(const Buffer& x)
{
    if(x.size() <= 10)
    {
        std::for_each(x.begin(), x.end(), [&](double i) { std::cout << i << ", "; });
    }
    else
    {
        std::for_each(x.begin(), x.begin() + 5, [&](double i) { std::cout << i << ", "; });
        std::cout << "..., ";
        std::for_each(x.end() - 5, x.end(), [&](double i) { std::cout << i << ", "; });
    }
    std::cout << std::endl;
}

template <class T>
struct check_all
{
    rtc::buffer<T> data{};
    bool operator()(const rtc::buffer<T>& x)
    {
        if(data.empty())
        {
            data = x;
            return true;
        }
        return allclose(data, x);
    }
};

template <class Solution>
auto report(const Solution& solution, bool pass)
{
    return test::make_predicate(solution.ToTemplateString(), [=] { return pass; });
}

struct Epilogue
{
    Epilogue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename D>
    __host__ __device__ constexpr void operator()(E& e, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>(ck::half_t& e,
                                                                          const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * e + beta_ * ck::type_convert<float>(d));
    }

    float alpha_;
    float beta_;
};
const std::string conv_compile_check = R"__ck__(
#include <${include}>

${template};

)__ck__";

TEST_CASE(test_problem_kernel)
{
    ck::host::conv::Problem_Conv_Fwd prob;
    prob.NumDim = 2;
    prob.G      = 32;
    prob.N      = 256;
    prob.C      = 32;
    prob.K      = 64;
    prob.Y      = 3;
    prob.X      = 3;
    prob.Hi     = 28;
    prob.Wi     = 28;
    prob.Ho     = 28;
    prob.Wo     = 28;
    check_all<ck::half_t> check;
    std::string epilogue = R"(
struct Epilogue
{
    __host__ __device__ Epilogue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename D>
    __host__ __device__ constexpr void operator()(E& e, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>(ck::half_t& e,
                                                                          const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * e + beta_ * ck::type_convert<float>(d));
    }

    float alpha_;
    float beta_;
};
)";
    std::string prologue = "";

    // length+stride arrays
    ck::Array<ck::index_t, 5> in_lengths{static_cast<int>(prob.G),
                                         static_cast<int>(prob.N),
                                         static_cast<int>(prob.C),
                                         static_cast<int>(prob.Hi),
                                         static_cast<int>(prob.Wi)};
    ck::Array<ck::index_t, 5> out_lengths{static_cast<int>(prob.G),
                                          static_cast<int>(prob.N),
                                          static_cast<int>(prob.K),
                                          static_cast<int>(prob.Ho),
                                          static_cast<int>(prob.Wo)};
    ck::Array<ck::index_t, 5> wei_lengths{static_cast<int>(prob.G),
                                          static_cast<int>(prob.K),
                                          static_cast<int>(prob.C),
                                          static_cast<int>(prob.Y),
                                          static_cast<int>(prob.X)};
    ck::Array<ck::index_t, 5> d_lengths = {};

    ck::Array<ck::index_t, 5> in_strides{static_cast<int>(prob.C),
                                         static_cast<int>(prob.Hi * prob.Wi * prob.G * prob.C),
                                         1,
                                         static_cast<int>(prob.Wi * prob.G * prob.C),
                                         static_cast<int>(prob.G * prob.C)};
    ck::Array<ck::index_t, 5> out_strides{static_cast<int>(prob.K),
                                          static_cast<int>(prob.Ho * prob.Wo * prob.G * prob.K),
                                          1,
                                          static_cast<int>(prob.Wo * prob.G * prob.K),
                                          static_cast<int>(prob.G * prob.K)};
    ck::Array<ck::index_t, 5> wei_strides{static_cast<int>(prob.K * prob.Y * prob.X * prob.C),
                                          static_cast<int>(prob.Y * prob.X * prob.C),
                                          1,
                                          static_cast<int>(prob.X * prob.C),
                                          static_cast<int>(prob.C)};
    ck::Array<ck::index_t, 5> d_strides = {};

    ck::Array<ck::index_t, 2> conv_filter_strides   = {1, 1};
    ck::Array<ck::index_t, 2> conv_filter_dilations = {1, 1};
    ck::Array<ck::index_t, 2> input_left_pads       = {1, 1};
    ck::Array<ck::index_t, 2> input_right_pads      = {1, 1};

    auto get_num_elems = [](const auto& tensor_lens) {
        return std::reduce(
            tensor_lens.begin(), tensor_lens.end(), 1, std::multiplies<ck::index_t>{});
    };

    auto in_dev  = to_gpu(generate_buffer<ck::half_t, ck::Array<ck::index_t, 5>>(
        get_num_elems(in_lengths), in_lengths, in_strides, 0));
    auto wei_dev = to_gpu(generate_buffer<ck::half_t, ck::Array<ck::index_t, 5>>(
        get_num_elems(wei_lengths), wei_lengths, wei_strides, 1));
    auto out_dev = to_gpu(generate_buffer<ck::half_t, ck::Array<ck::index_t, 5>>(
        get_num_elems(out_lengths), out_lengths, out_strides, 2));

    // CK Verficiation: Reference Kernel
    /**bool pass = true;
    Tensor<ck::half_t> in_host(in_lengths, in_strides);
    in_host.GenerateTensorValue(GeneratorTensor_1<ck::half_t>{1});
    Tensor<ck::half_t> wei_host(wei_lengths, wei_strides);
    wei_host.GenerateTensorValue(GeneratorTensor_1<ck::half_t>{1});
    Tensor<ck::half_t> out_host(out_lengths, out_strides);

    std::vector<ck::index_t> conv_filter_strides_   = {1, 1};
    std::vector<ck::index_t> conv_filter_dilations_ = {1, 1};
    std::vector<ck::index_t> input_left_pads_       = {1, 1};
    std::vector<ck::index_t> input_right_pads_      = {1, 1};

    auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<
        2,
        ck::half_t,
        ck::half_t,
        ck::half_t,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        Epilogue>();

    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(in_host,
                                              wei_host,
                                              out_host,
                                              conv_filter_strides_,
                                              conv_filter_dilations_,
                                              input_left_pads_,
                                              input_right_pads_,
                                              ck::tensor_operation::element_wise::PassThrough{},
                                              ck::tensor_operation::element_wise::PassThrough{},
                                              Epilogue{1.0f, 1.0f});
    out_host.SetZero();
    ref_invoker.Run(ref_argument);**/

    for(auto solution : prob.GetSolutions("gfx908", prologue, epilogue))
    {
        auto src = ck::host::InterpolateString(
            conv_compile_check,
            {{"include",
              "ck/tensor_operation/gpu/device/impl/"
              "copy_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"},
             {"template", solution.ToTemplateString()}});

        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        auto name           = solution.GetTemplateParameter<std::string>("name");
        options.kernel_name = "run_" + name;
        auto k              = rtc::compile_kernel(srcs, options);

        auto block_size = solution.GetTemplateParameter<ck::index_t>("BlockSize");

        // Grid size calculation
        auto tmp = get_launch_params(solution, out_lengths, out_strides);

        auto grid_size = tmp * in_lengths[1];

        k.launch(nullptr, grid_size * block_size, block_size)(in_dev.data(),
                                                              wei_dev.data(),
                                                              out_dev.data(),
                                                              in_lengths,
                                                              in_strides,
                                                              wei_lengths,
                                                              wei_strides,
                                                              out_lengths,
                                                              out_strides,
                                                              conv_filter_strides,
                                                              conv_filter_dilations,
                                                              input_left_pads,
                                                              input_right_pads);

        // auto res = rtc::from_gpu(out_dev);
        // pass &= ck::utils::check_err(res, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
        // assert(pass);
        CHECK(report(solution, check(rtc::from_gpu(out_dev))));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
