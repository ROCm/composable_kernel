#include "ck/host/device_gemm_multiple_d/problem.hpp"
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/copy_conv_fwd_op.hpp"
#include "ck/host/device_grouped_conv_fwd_multiple_d/copy_conv_fwd_problem.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/utils.hpp"
#include "ck/tensor_operation/gpu/device/impl/mod_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/library/utility/host_tensor.hpp"
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
    Prologue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

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
using DeviceConv = ck::tensor_operation::device::ModDeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
      2, 
      ck::tensor_layout::convolution::GNHWC, 
      ck::tensor_layout::convolution::GKYXC, 
      ck::Tuple<>, ck::tensor_layout::convolution::GNHWK, 
      ck::half_t, ck::half_t, float, ck::half_t, ck::Tuple<>, ck::half_t, 
      ck::tensor_operation::element_wise::PassThrough, 
      ck::tensor_operation::element_wise::PassThrough, Prologue, 
      ck::tensor_operation::device::ConvolutionForwardSpecialization::Default, 
      ck::tensor_operation::device::GemmSpecialization::MNPadding, 
      1, 256, 128, 256, 32, 8, 8, 32, 32, 2, 4, 
      ck::Sequence<4, 64, 1>, ck::Sequence<1, 0, 2>, ck::Sequence<1, 0, 2>, 
      2, 8, 8, 1, ck::Sequence<4, 64, 1>, ck::Sequence<1, 0, 2>, 
      ck::Sequence<1, 0, 2>, 2, 8, 8, 1, 1, 1, ck::Sequence<1, 32, 1, 8>, 8>;

    constexpr ck::index_t NumATensor = ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();
    constexpr ck::index_t NumBTensor = ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();

extern "C" __global__ void kernel_group_conv_fwd(
    const ck::half_t* p_as_grid,
    const ck::half_t* p_bs_grid,
    DeviceConv::GridwiseGemm::DsGridPointer p_ds_grid,
    ck::half_t* __restrict__ p_e_grid,
    const ck::tensor_operation::element_wise::PassThrough a_element_op,
    const ck::tensor_operation::element_wise::PassThrough b_element_op,
    const Prologue cde_element_op,
    const ck::index_t batch_count,
    const DeviceConv::AGridDesc_AK0_M_AK1 a_grid_desc_k0_m_k1,
    const DeviceConv::BGridDesc_BK0_N_BK1 b_grid_desc_k0_n_k1,
    const DeviceConv::DsGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
        ds_grid_desc_mblock_mperblock_nblock_nperblock,
    const DeviceConv::EGridDesc_MBlock_MPerBlock_NBlock_NPerBlock
        e_grid_desc_mblock_mperblock_nblock_nperblock_,
    const DeviceConv::Block2ETileMap block_2_ctile_map,
		const ck::tensor_operation::device::ComputePtrOffsetOfStridedBatch<NumATensor, NumBTensor, 0> compute_ptr_offset_of_batch) {

    using CDEElementOp = Prologue;

    using GridwiseGemm = DeviceConv::GridwiseGemm;

    ck::tensor_operation::device::mod_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle<
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
                  p_as_grid,
                  p_bs_grid,
                  p_ds_grid,
                  p_e_grid,
                  a_element_op,
                  b_element_op,
                  cde_element_op,
                  batch_count,
                  a_grid_desc_k0_m_k1,
                  b_grid_desc_k0_n_k1,
                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                  e_grid_desc_mblock_mperblock_nblock_nperblock_,
                  block_2_ctile_map,
                  compute_ptr_offset_of_batch);
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
)";

    std::string epilogue = "";

    using CDEElementOp = Prologue;

    using DeviceConv =
        ck::tensor_operation::device::ModDeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
            2,
            ck::tensor_layout::convolution::GNHWC,
            ck::tensor_layout::convolution::GKYXC,
            ck::Tuple<>,
            ck::tensor_layout::convolution::GNHWK,
            ck::half_t,
            ck::half_t,
            float,
            ck::half_t,
            ck::Tuple<>,
            ck::half_t,
            ck::tensor_operation::element_wise::PassThrough,
            ck::tensor_operation::element_wise::PassThrough,
            CDEElementOp,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
            ck::tensor_operation::device::GemmSpecialization::MNKPadding,
            1,
            256,
            128,
            256,
            32,
            8,
            8,
            32,
            32,
            4,
            2,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            ck::Sequence<4, 64, 1>,
            ck::Sequence<1, 0, 2>,
            ck::Sequence<1, 0, 2>,
            2,
            8,
            8,
            1,
            1,
            1,
            ck::Sequence<1, 32, 1, 8>,
            8>;

    // length+stride arrays
    std::array<ck::index_t, 5> in_lengths{static_cast<int>(prob.G),
                                          static_cast<int>(prob.N),
                                          static_cast<int>(prob.C),
                                          static_cast<int>(prob.Hi),
                                          static_cast<int>(prob.Wi)};
    std::array<ck::index_t, 5> out_lengths{static_cast<int>(prob.G),
                                           static_cast<int>(prob.N),
                                           static_cast<int>(prob.K),
                                           static_cast<int>(prob.Ho),
                                           static_cast<int>(prob.Wo)};
    std::array<ck::index_t, 5> wei_lengths{static_cast<int>(prob.G),
                                           static_cast<int>(prob.K),
                                           static_cast<int>(prob.C),
                                           static_cast<int>(prob.Y),
                                           static_cast<int>(prob.X)};
    std::array<ck::index_t, 5> d_lengths = {};

    std::array<ck::index_t, 5> in_strides{123887616,
                                          static_cast<int>(prob.Hi * prob.Wi * prob.G * prob.C),
                                          1,
                                          static_cast<int>(prob.Wi * prob.G * prob.C),
                                          static_cast<int>(prob.G * prob.C)};
    std::array<ck::index_t, 5> out_strides{42467328,
                                           static_cast<int>(prob.Ho * prob.Wo * prob.G * prob.K),
                                           1,
                                           static_cast<int>(prob.Wo * prob.G * prob.K),

                                           static_cast<int>(prob.G * prob.K)};
    std::array<ck::index_t, 5> wei_strides{442368,
                                           static_cast<int>(prob.Y * prob.X * prob.C),
                                           1,
                                           static_cast<int>(prob.X * prob.C),
                                           static_cast<int>(prob.C)};
    std::array<ck::index_t, 5> d_strides = {};

    std::array<ck::index_t, 2> conv_filter_strides   = {2, 2};
    std::vector<ck::index_t> conv_filter_strides_    = {2, 2};
    std::array<ck::index_t, 2> conv_filter_dilations = {1, 1};
    std::vector<ck::index_t> conv_filter_dilations_  = {1, 1};
    std::array<ck::index_t, 2> input_left_pads       = {1, 1};
    std::vector<ck::index_t> input_left_pads_        = {1, 1};
    std::array<ck::index_t, 2> input_right_pads      = {1, 1};
    std::vector<ck::index_t> input_right_pads_       = {1, 1};

    auto get_num_elems = [](const auto& tensor_lens) {
        return std::reduce(
            tensor_lens.begin(), tensor_lens.end(), 1, std::multiplies<ck::index_t>{});
    };

    auto in_dev  = to_gpu(generate_buffer<ck::half_t>(get_num_elems(in_lengths), 0));
    auto wei_dev = to_gpu(generate_buffer<ck::half_t>(get_num_elems(wei_lengths), 1));
    auto out_dev = to_gpu(generate_buffer<ck::half_t>(get_num_elems(out_lengths), 2));

    // populated arg call
    auto arg = DeviceConv::Argument(in_dev.data(),
                                    wei_dev.data(),
                                    std::array<const void*, 0>{},
                                    out_dev.data(),
                                    in_lengths,
                                    in_strides,
                                    wei_lengths,
                                    wei_strides,
                                    std::array<std::array<ck::index_t, 5>, 0>{},
                                    std::array<std::array<ck::index_t, 5>, 0>{},
                                    out_lengths,
                                    out_strides,
                                    conv_filter_strides,
                                    conv_filter_dilations,
                                    input_left_pads,
                                    input_right_pads,
                                    ck::tensor_operation::element_wise::PassThrough{},
                                    ck::tensor_operation::element_wise::PassThrough{},
                                    CDEElementOp{1.0f, 1.0f});

    constexpr ck::index_t NumATensor =
        ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();
    constexpr ck::index_t NumBTensor =
        ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();

    // Amber: removed const because compiler makes it an r-value
    auto as_grid_desc_ak0_m_ak1 =
        generate_tuple([&](auto) { return arg.a_grid_desc_ak0_m_ak1_; }, ck::Number<NumATensor>{});
    auto bs_grid_desc_bk0_n_bk1 =
        generate_tuple([&](auto) { return arg.b_grid_desc_bk0_n_bk1_; }, ck::Number<NumBTensor>{});

    for(auto solution : prob.GetSolutions("gfx908", prologue, epilogue))
    {
        auto src =
            ck::host::InterpolateString(conv_compile_check, {{"include", prob.GetIncludeHeader()}});

        std::ofstream ofh("kernel.txt");
        ofh << src;

        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        options.kernel_name = "kernel_group_conv_fwd";
        auto k              = rtc::compile_kernel(srcs, options);

        auto grid_size =
            arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_) * arg.num_group_;
        auto block_size = solution.GetTemplateParameter<ck::index_t>("BlockSize");
        ;

        ofh.close();
        // print arg kernels - host_side
        // arg.Print();

        k.launch(nullptr, grid_size * block_size, block_size)(
            arg.p_as_grid_,
            arg.p_bs_grid_,
            arg.p_ds_grid_,
            arg.p_e_grid_,
            arg.a_element_op_,
            arg.b_element_op_,
            arg.cde_element_op_,
            arg.a_g_n_c_wis_lengths_[0], // Group count
            as_grid_desc_ak0_m_ak1,
            bs_grid_desc_bk0_n_bk1,
            arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
            arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
            arg.block_2_etile_map_,
            arg.compute_ptr_offset_of_batch_);

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
            CDEElementOp>();

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
                                                  CDEElementOp{1.0f, 1.0f});
        // std::cout << "Ref args" << std::endl;
        // ref_argument.Print();

        ref_invoker.Run(ref_argument);

        bool pass = true;
        auto res  = rtc::from_gpu(out_dev);
        std::ofstream ofh2("res.txt");
        pass &= ck::utils::check_err(res, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);
        for(int i = 0; i < res.size(); i++)
        {
            auto tmp = (res.data())[i];
            ofh2 << std::to_string(static_cast<int>(tmp)) << ", ";
        }
        ofh2.close();
        assert(pass);**/
        auto res = rtc::from_gpu(out_dev);
        CHECK(report(solution, check(res)));
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
