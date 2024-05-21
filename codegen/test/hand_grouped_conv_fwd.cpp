#include "ck/host/device_grouped_conv_fwd_multiple_d/copy_conv_fwd_op.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/copy_conv_fwd_problem.hpp"
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

template <class T>
rtc::buffer<T> generate_buffer(std::size_t n, std::size_t seed = 0)
{
    rtc::buffer<T> result(n);
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

struct Prologue
{
    Prologue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

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

struct Prologue
{
    __host__ __device__ Prologue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename D>
    __host__ __device__ constexpr void operator()(E& e, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, ck::half_t>(
        ck::half_t& e, const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * e + beta_ * ck::type_convert<float>(d));
    }

    float alpha_;
    float beta_;
};

extern "C" __global__ void kernel_group_conv_fwd(
    const ck::half_t* in_dev,
    const ck::half_t* wei_dev,
    ck::half_t* __restrict__ out_dev,
    ck::Array<ck::index_t, 5> in_lengths,
    ck::Array<ck::index_t, 5> in_strides,
    ck::Array<ck::index_t, 5> wei_lengths,
    ck::Array<ck::index_t, 5> wei_strides,
    ck::Array<ck::index_t, 5> out_lengths,
    ck::Array<ck::index_t, 5> out_strides,
    ck::Array<ck::index_t, 2> conv_filter_strides, 
    ck::Array<ck::index_t, 2> conv_filter_dilations, 
    ck::Array<ck::index_t, 2> input_left_pads, 
    ck::Array<ck::index_t, 2> input_right_pads, 
    const ck::tensor_operation::element_wise::PassThrough a_element_op,
    const ck::tensor_operation::element_wise::PassThrough b_element_op,
    const Prologue cde_element_op
    ) {


    using DeviceConv = ck::tensor_operation::device::CopyDeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
          2, 
          ck::tensor_layout::convolution::NHWGC, 
          ck::tensor_layout::convolution::GKYXC, 
          ck::Tuple<>, ck::tensor_layout::convolution::NHWGK, 
          ck::half_t, ck::half_t, float, ck::half_t, ck::Tuple<>, ck::half_t, 
          ck::tensor_operation::element_wise::PassThrough, 
          ck::tensor_operation::element_wise::PassThrough, Prologue, 
          ck::tensor_operation::device::ConvolutionForwardSpecialization::Default, 
          ck::tensor_operation::device::GemmSpecialization::MNKPadding, 
          1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4, 
          ck::Sequence<4, 64, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 
          2, 8, 8, 1, ck::Sequence<4, 64, 1>, ck::Sequence<1, 0, 2>, 
          ck::Sequence<1, 0, 2>, 2, 8, 8, 1, 1, 1, ck::Sequence<1, 32, 1, 8>, 8>;

    // populated arg call
    auto arg = DeviceConv::Argument(in_dev,
                                    wei_dev,
                                    ck::Array<const void*, 0>{},
                                    out_dev,
                                    in_lengths,
                                    in_strides,
                                    wei_lengths,
                                    wei_strides,
                                    ck::Array<ck::Array<ck::index_t, 5>, 0>{},
                                    ck::Array<ck::Array<ck::index_t, 5>, 0>{},
                                    out_lengths,
                                    out_strides,
                                    conv_filter_strides,
                                    conv_filter_dilations,
                                    input_left_pads,
                                    input_right_pads,
                                    ck::tensor_operation::element_wise::PassThrough{},
                                    ck::tensor_operation::element_wise::PassThrough{},
                                    Prologue{1.0f, 1.0f});


    using GridwiseGemm = DeviceConv::GridwiseGemm;

    constexpr ck::index_t NumATensor = ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();
    constexpr ck::index_t NumBTensor = ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();

    static constexpr auto I0 = ck::Number<0>{};

    ck::tensor_operation::device::copy_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
                    GridwiseGemm,
                    const ck::half_t*,
                    const ck::half_t*,
                    typename GridwiseGemm::DsGridPointer,
                    ck::half_t,
                    ck::tensor_operation::element_wise::PassThrough,
                    ck::tensor_operation::element_wise::PassThrough,
                    Prologue,
                    DeviceConv::AGridDesc_AK0_M_AK1,
                    DeviceConv::BGridDesc_BK0_N_BK1,
                    DeviceConv::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceConv::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock,
                    DeviceConv::Block2ETileMap,
		                ck::tensor_operation::device::ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, 0>,

                    // TODO(Amber): double check these bool flags
                    ck::integral_constant<bool, true>{}, // HasMainKBlockLoop
                    false, // isMultiA
                    false> // isMultiB
                  (
                    arg.p_as_grid_.At(I0),
                    arg.p_bs_grid_.At(I0),
                    arg.p_ds_grid_,
                    arg.p_e_grid_,
                    arg.a_element_op_,
                    arg.b_element_op_,
                    arg.cde_element_op_,
                    arg.a_g_n_c_wis_lengths_[0], // Group count
                    arg.a_grid_desc_ak0_m_ak1_,
                    arg.b_grid_desc_bk0_n_bk1_,
                    arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                    arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                    arg.block_2_etile_map_,
                    arg.compute_ptr_offset_of_batch_);
}

)__ck__";

TEST_CASE(test_problem_kernel)
{
    ck::host::conv::Copy_Problem_Conv_Fwd prob;
    prob.G  = 1;
    prob.N  = 128;
    prob.C  = 192;
    prob.K  = 256;
    prob.Y  = 3;
    prob.X  = 3;
    prob.Hi = 71;
    prob.Wi = 71;
    prob.Ho = 36;
    prob.Wo = 36;
    check_all<ck::half_t> check;
    std::string prologue = R"(
struct Prologue
{
    __host_- __device__ Prologue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

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

    std::string epilogue = "";

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

    ck::Array<ck::index_t, 5> in_strides{123887616,
                                         static_cast<int>(prob.Hi * prob.Wi * prob.G * prob.C),
                                         1,
                                         static_cast<int>(prob.Wi * prob.G * prob.C),
                                         static_cast<int>(prob.G * prob.C)};
    ck::Array<ck::index_t, 5> out_strides{42467328,
                                          static_cast<int>(prob.Ho * prob.Wo * prob.G * prob.K),
                                          1,
                                          static_cast<int>(prob.Wo * prob.G * prob.K),

                                          static_cast<int>(prob.G * prob.K)};
    ck::Array<ck::index_t, 5> wei_strides{442368,
                                          static_cast<int>(prob.Y * prob.X * prob.C),
                                          1,
                                          static_cast<int>(prob.X * prob.C),
                                          static_cast<int>(prob.C)};
    ck::Array<ck::index_t, 5> d_strides = {};

    ck::Array<ck::index_t, 2> conv_filter_strides   = {2, 2};
    std::vector<ck::index_t> conv_filter_strides_   = {2, 2};
    ck::Array<ck::index_t, 2> conv_filter_dilations = {1, 1};
    std::vector<ck::index_t> conv_filter_dilations_ = {1, 1};
    ck::Array<ck::index_t, 2> input_left_pads       = {1, 1};
    std::vector<ck::index_t> input_left_pads_       = {1, 1};
    ck::Array<ck::index_t, 2> input_right_pads      = {1, 1};
    std::vector<ck::index_t> input_right_pads_      = {1, 1};

    auto get_num_elems = [](const auto& tensor_lens) {
        return std::reduce(
            tensor_lens.begin(), tensor_lens.end(), 1, std::multiplies<ck::index_t>{});
    };

    auto in_dev  = to_gpu(generate_buffer<ck::half_t>(get_num_elems(in_lengths), 0));
    auto wei_dev = to_gpu(generate_buffer<ck::half_t>(get_num_elems(wei_lengths), 1));
    auto out_dev = to_gpu(generate_buffer<ck::half_t>(get_num_elems(out_lengths), 2));

    for(auto solution : prob.GetSolutions("gfx908", prologue, epilogue))
    {
        auto src = ck::host::InterpolateString(
            conv_compile_check,
            {{"include",
              "ck/tensor_operation/gpu/device/impl/"
              "copy_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"}});

        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        options.kernel_name = "kernel_group_conv_fwd";
        auto k              = rtc::compile_kernel(srcs, options);

        auto block_size = solution.GetTemplateParameter<ck::index_t>("BlockSize");

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

        // Validation: CK Reference Kernel
        /**Tensor<ck::half_t> in_host(in_lengths, in_strides);
        in_host.GenerateTensorValue(GeneratorTensor_1<ck::half_t>{1});
        Tensor<ck::half_t> wei_host(wei_lengths, wei_strides);
        wei_host.GenerateTensorValue(GeneratorTensor_1<ck::half_t>{1});
        Tensor<ck::half_t> out_host(out_lengths, out_strides);

        auto ref_conv = ck::tensor_operation::host::ReferenceConvFwd<
            2,
            ck::half_t,
            ck::half_t,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            Prologue>();

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
                                                  Prologue{1.0f, 1.0f});

        ref_invoker.Run(ref_argument);

        bool pass = true;
        auto res  = rtc::from_gpu(out_dev);
        pass &= ck::utils::check_err(res, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);

        if(pass)
        {
            std::cout << "%%%%%%%%%%%%%%% VERIFICATION PASSED! %%%%%%%%%%%%%\n";
        }
        else
        {
            std::cout << "!!!!!!!!!!!! ERROR !!!!!!!!!!!\n";
            std::abort();
        }**/

        auto res = rtc::from_gpu(out_dev);
        CHECK(report(solution, check(res)));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
