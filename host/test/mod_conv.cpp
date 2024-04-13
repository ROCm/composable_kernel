#include "ck/host/device_gemm_multiple_d/problem.hpp"
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/conv/conv_op.hpp"
#include "ck/host/conv/dev_conv.hpp"
#include "ck/host/headers.hpp"
#include "ck/host/stringutils.hpp"
#include "ck/host/types.hpp"
#include "ck/host/utils.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/tensor_operation/gpu/device/impl/copy_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/copy_transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/copy_matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/library/utility/iterator.hpp"
#include "ck/library/utility/algorithm.hpp"
#include "ck/library/utility/ranges.hpp"
#include "ck/library/utility/check_err.hpp"
//#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"
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
#include <variant>
#include <any>

// using half = _Float16;
// using half = __fp16;

std::vector<rtc::src_file> get_headers_for_test()
{
    std::vector<rtc::src_file> result;
    auto hs = ck::host::GetHeaders();
    std::transform(
        hs.begin(), hs.end(), std::back_inserter(result), [&](const auto& p) -> rtc::src_file {
            auto s = p.second;
            std::string content{s.first, s.second};
            return {p.first, content};
        });
    return result;
}

template <class T>
rtc::buffer<T> generate_buffer(std::size_t n, std::size_t seed = 0)
{
    rtc::buffer<T> result(n);
    /**std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dis(-1.0);
    std::generate(result.begin(), result.end(), [&] { return dis(gen); });**/
    std::fill(result.begin(), result.end(), 1);
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

using layouts = std::variant<ck::tensor_layout::convolution::GNHWC,
                             ck::tensor_layout::convolution::GNHWK,
                             ck::tensor_layout::convolution::GKYXC>;
layouts layout_type(std::string type)
{
    if(type == "ck::tensor_layout::convolution::GNHWC")
    {
        return ck::tensor_layout::convolution::GNHWC{};
    }
    else if(type == "ck::tensor_layout::convolution::GNHWK")
    {
        return ck::tensor_layout::convolution::GNHWK{};
    }
    else if(type == "ck::tensor_layout::convolution::GKYXC")
    {
        return ck::tensor_layout::convolution::GKYXC{};
    }
    return ck::tensor_layout::convolution::GNHWC{};
}

// method to check GemmType
ck::tensor_operation::device::GemmSpecialization gemm_type(std::string type)
{
    if(type == "ck::tensor_operation::device::GemmSpecialization::Default")
    {
        return ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    }
    return ck::tensor_operation::device::GemmSpecialization::Default;
}

// TODO: edit/repurpose these to instantiate structs then call the wrapper class instead
template <typename CDesc_MRaw_NRaw>
// ck::tensor_operation::device::Padder
auto pad(ck::index_t mpb,
         ck::index_t npb,
         ck::index_t kpb,
         ck::tensor_operation::device::GemmSpecialization gemm,
         CDesc_MRaw_NRaw conv)
{
    ck::tensor_operation::device::CopyMatrixPadder<
        ck::tensor_operation::device::GemmSpecialization::MNKPadding,
        ck::index_t,
        ck::index_t,
        ck::index_t>
        a;
    a.MPerTile_ = mpb;
    a.NPerTile_ = npb;
    a.KPerTile_ = kpb;
    auto res    = ck::tensor_operation::device::Padder(a, conv);
    return res.grid_desc(a, conv);
}
// ck::tensor_operation::TransformConv
auto transform_conv(ck::index_t num_dim,
                    ck::tensor_operation::device::ConvolutionForwardSpecialization spec,
                    layouts e_layout,
                    ck::Array<ck::index_t, 5> out_lengths,
                    ck::Array<ck::index_t, 5> out_strides)
{
    if(num_dim == 2 &&
       spec == ck::tensor_operation::device::ConvolutionForwardSpecialization::Default)
    {
        ck::tensor_operation::TransformConvFwdToGemm<
            2,
            ck::tensor_operation::device::ConvolutionForwardSpecialization::Default>
            conv_fwd;
        // return conv_fwd.template MakeCDescriptor_M_N<ck::tensor_layout::convolution::GNHWK>(
        // out_lengths, out_strides);

        auto res =
            ck::tensor_operation::TransformConv(e_layout, out_lengths, out_strides, conv_fwd);
        return res.transform_func(e_layout, out_lengths, out_strides, conv_fwd);
    }
}

ck::tensor_operation::device::ConvolutionForwardSpecialization conv_type(std::string type)
{
    if(type == "ck::tensor_operation::device::ConvolutionForwardSpecialization::Default")
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
    }
    return ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
}

template <typename CGridDesc_M_N>
auto block_2_etile(ck::index_t m_per_block, ck::index_t n_per_block, CGridDesc_M_N matrix_padder)
{
    // TODO: figure out how to pass parameters properly -> not scalable for this method
    if(m_per_block == 128 && n_per_block == 256)
    {
        ck::BlockToCTileMap_M00_N0_M01Adapt<128, 256, CGridDesc_M_N> b2e(matrix_padder);
        return b2e.CalculateGridSize(matrix_padder);
    }
    // b2e	= ck::BlockToCTileMap_M00_N0_M01Adapt(m_per_block, n_per_block);
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

// TODO(Amber): remove as Prologue will be merged later on
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
// TODO(Amber): fix parameters
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
          ck::tensor_layout::convolution::GNHWC, 
          ck::tensor_layout::convolution::GKYXC, 
          ck::Tuple<>, ck::tensor_layout::convolution::GNHWK, 
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

    auto as_grid_desc_ak0_m_ak1 =
        generate_tuple([&](auto) { return arg.a_grid_desc_ak0_m_ak1_; }, ck::Number<NumATensor>{});
    auto bs_grid_desc_bk0_n_bk1 =
        generate_tuple([&](auto) { return arg.b_grid_desc_bk0_n_bk1_; }, ck::Number<NumBTensor>{});

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
    ck::host::conv::Problem_Conv prob;
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

    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};
    static constexpr auto I3 = ck::Number<3>{};

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

    /*
    auto in_tmp = get_num_elems(in_lengths);
    std::cout << std::to_string(in_tmp) << std::endl;
    std::cout << "Lengths" << std::endl;
    std::cout << in_lengths << std::endl;
    std::cout << wei_lengths << std::endl;
    std::cout << out_lengths << std::endl;
    std::cout << "Strides" << std::endl;
    std::cout << in_strides << std::endl;
    std::cout << wei_strides << std::endl;
    std::cout << out_strides << std::endl;
    */

    auto in_dev  = to_gpu(generate_buffer<ck::half_t>(get_num_elems(in_lengths), 0));
    auto wei_dev = to_gpu(generate_buffer<ck::half_t>(get_num_elems(wei_lengths), 1));
    auto out_dev = to_gpu(generate_buffer<ck::half_t>(get_num_elems(out_lengths), 2));

    auto out = generate_buffer<ck::half_t>(get_num_elems(out_lengths), 2);
    auto in  = generate_buffer<ck::half_t>(get_num_elems(in_lengths), 0);
    auto wei = generate_buffer<ck::half_t>(get_num_elems(wei_lengths), 1);
    std::cout << "Sizes: " << get_num_elems(in_lengths) << ", " << get_num_elems(wei_lengths)
              << ", " << get_num_elems(out_lengths) << std::endl;
    std::cout << "buf sizes" << std::endl;
    std::cout << out.size() << std::endl;
    std::cout << in.size() << std::endl;
    std::cout << wei.size() << std::endl;

    // constexpr ck::index_t NumATensor =
    //    ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();
    // constexpr ck::index_t NumBTensor =
    //    ck::tensor_operation::device::GetNumABTensors<false, ck::half_t>();

    // Amber: removed const because compiler makes it an r-value
    // auto as_grid_desc_ak0_m_ak1 =
    //  generate_tuple([&](auto) { return arg.a_grid_desc_ak0_m_ak1_; }, ck::Number<NumATensor>{});
    // auto bs_grid_desc_bk0_n_bk1 =
    //  generate_tuple([&](auto) { return arg.b_grid_desc_bk0_n_bk1_; }, ck::Number<NumBTensor>{});

    for(auto solution : prob.GetSolutions("gfx908", prologue, epilogue))
    {
        auto src = ck::host::InterpolateString(
            conv_compile_check,
            {{"include",
              "ck/tensor_operation/gpu/device/impl/"
              "copy_device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"}});

        std::ofstream ofh("kernel.txt");
        ofh << src;

        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        options.kernel_name = "kernel_group_conv_fwd";
        auto k              = rtc::compile_kernel(srcs, options);

        auto m_per_block = solution.GetTemplateParameter<ck::index_t>("MPerBlock");
        auto n_per_block = solution.GetTemplateParameter<ck::index_t>("NPerBlock");
        auto k_per_block = solution.GetTemplateParameter<ck::index_t>("KPerBlock");
        auto num_dim     = solution.GetTemplateParameter<ck::index_t>("NumDim");
        auto block_size  = solution.GetTemplateParameter<ck::index_t>("BlockSize");
        auto GemmType    = solution.GetTemplateParameter<std::string>("GemmSpecialization");
        auto ConvType    = solution.GetTemplateParameter<std::string>("ConvSpecialization");
        auto out_layout  = solution.GetTemplateParameter<std::string>("LayoutE");
        ck::tensor_operation::device::GemmSpecialization GemmSpec = gemm_type(GemmType);
        ck::tensor_operation::device::ConvolutionForwardSpecialization ConvSpec =
            conv_type(ConvType);
        layouts ELayout = layout_type(out_layout);

        // TODO: replace with repurposed factory function calls
        auto conv_to_gemm_transformer =
            transform_conv(num_dim, ConvSpec, ELayout, out_lengths, out_strides);
        // decltype(conv_to_gemm_transformer)::foo = 1;
        // auto conv_to_gemm_transformer = ck::tensor_operation::TransformConvFwdToGemm<
        //  2,
        // ck::tensor_operation::device::ConvolutionForwardSpecialization::Default>{};

        auto matrix_padder =
            pad(m_per_block, n_per_block, k_per_block, GemmSpec, conv_to_gemm_transformer);
        /**auto matrix_padder = ck::tensor_operation::device::MatrixPadder<
            ck::tensor_operation::device::GemmSpecialization::MNKPadding,
            ck::index_t,
            ck::index_t,
            ck::index_t>{static_cast<int>(m_per_block),
                         static_cast<int>(n_per_block),
                         static_cast<int>(k_per_block)};**/

        auto b2e = block_2_etile(m_per_block, n_per_block, matrix_padder);
        // ck::BlockToCTileMap_M00_N0_M01Adapt<m_per_block, n_per_block,
        // decltype(matrix_padder)>(matrix_padder);

        // E grid desc + block 2 etile: use method implementations without calling them
        const ck::index_t N = out_lengths[1];
        const ck::index_t K = out_lengths[2];

        const auto KStride         = I1;
        const ck::index_t WoStride = out_strides[num_dim + 2];

        const ck::index_t NHoWo = N * ck::accumulate_n<ck::index_t>(
                                          out_lengths.begin() + 3, num_dim, 1, std::multiplies<>());

        const auto out_gemmm_gemmn_desc = make_naive_tensor_descriptor(
            ck::make_tuple(NHoWo, K), ck::make_tuple(WoStride, KStride));
        // hard-code PadM/N = true for now: MNK Padding
        auto out = ck::tensor_operation::device::PadTensorDescriptor(
            out_gemmm_gemmn_desc,
            ck::make_tuple(m_per_block, n_per_block),
            ck::Sequence<true, true>{});

        // Grid size calculation
        const ck::index_t M0 = ck::math::integer_divide_ceil(out.GetLength(I0), m_per_block);
        const ck::index_t N0 = ck::math::integer_divide_ceil(out.GetLength(I1), n_per_block);

        auto grid_size = b2e * in_lengths[1];

        // ofh << "Grid Size: " << grid_size << std::endl;
        // ofh << "Block Size: " << block_size << std::endl;
        ofh.close();
        // print arg kernels - host_side
        // arg.Print();
        std::cout << "launched" << std::endl;

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
        std::cout << "Ref args" << std::endl;
        ref_argument.Print();

        ref_invoker.Run(ref_argument);

        bool pass = true;
        auto res  = rtc::from_gpu(out_dev);
        // LogRangeAsType<float>(std::cout << "out  : ", out_host.mData, ", ") << std::endl;
        std::ofstream ofh2("res.txt");
        pass &= ck::utils::check_err(res, out_host, "Error: incorrect results!", 1e-5f, 1e-4f);

        // ofh2 << "Check: " << pass << std::endl;
        // ofh2 << res.size() << std::endl;
        // for(int i = 0; i < res.size(); i++)
        //{
        //    auto tmp = (res.data())[i];
        //    ofh2 << std::to_string(static_cast<int>(tmp)) << ", ";
        //}
        // ofh2.close();

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
