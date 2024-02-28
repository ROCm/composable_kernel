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
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/io.hpp"
//#include "ck/host/tuples.hpp"
//#include "ck/host/seq.hpp"
//#include "ck/host/tensor_desc.hpp"
//#include "ck/host/transform.hpp"
#include <algorithm>
#include <cmath>
#include <iterator>
#include <numeric>
#include <random>
#include <test.hpp>
#include <rtc/compile_kernel.hpp>
#include <rtc/hip.hpp>
#include <fstream>

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

/**#define ToLayout(lay_out,x){ \
	if(lay_out == "ck::tensor_layout::gemm::RowMajor"){\
		ck::tensor_layout::gemm::RowMajor layout ;\
		x = layout;\
	}else{\
		ck::tensor_layout::gemm::ColumnMajor layout ;\
		x = layout;\
	}\
}**/
/**auto to_layout(std::string layout){
	if(layout == "ck::tensor_layout::gemm::RowMajor"){
		return ck::tensor_layout::gemm::RowMajor{};
	}else{
		return ck::tensor_layout::gemm::ColumnMajor{};
	}
	return -1;
}**/

const std::string conv_compile_check = R"__ck__(
#include <${include}>

${template};

)__ck__";

TEST_CASE(test_problem_kernel)
{
    ck::host::conv::Problem_Conv prob;
    prob.G  = 4;
    prob.N  = 64;
    prob.C  = 32;
    prob.K  = 32;
    prob.Y  = 3;
    prob.X  = 3;
    prob.Hi = 32;
    prob.Wi = 32;
    prob.Ho = 32;
    prob.Wo = 32;
    check_all<half> check;
    auto a               = to_gpu(generate_buffer<half>(64 * 64, 0));
    auto b               = to_gpu(generate_buffer<half>(64 * 64, 1));
    auto c               = to_gpu(generate_buffer<half>(64 * 64, 2));
    std::string prologue = R"(struct Prologue
{
    Prologue(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename E, typename C, typename D>
    __host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;

    template <>
    __host__ __device__ constexpr void operator()<ck::half_t, float, ck::half_t>(
        ck::half_t& e, const float& c, const ck::half_t& d) const
    {
        e = ck::type_convert<ck::half_t>(alpha_ * c + beta_ * ck::type_convert<float>(d));
    };

    float alpha_;
    float beta_;
};
)";
    std::string epilogue = "";

    static constexpr auto I0 = ck::Number<0>{};
    static constexpr auto I1 = ck::Number<1>{};
    static constexpr auto I2 = ck::Number<2>{};
    static constexpr auto I3 = ck::Number<3>{};

    // static constexpr auto I1 = Number<1>{};
    // length+stride arrays
    std::array<std::size_t, 5> in_lengths{prob.G, prob.N, prob.C, prob.Hi, prob.Wi};
    std::array<std::size_t, 5> out_lengths{prob.G, prob.N, prob.K, prob.Ho, prob.Wo};
    std::array<std::size_t, 5> wei_lengths{prob.G, prob.K, prob.C, prob.Y, prob.X};
    std::array<std::size_t, 5> d_lengths = {};

    std::array<std::size_t, 5> in_strides{
        prob.C, prob.Hi * prob.Wi * prob.G * prob.C, 1, prob.Wi * prob.G * prob.C, prob.G * prob.C};
    std::array<std::size_t, 5> out_strides{
        prob.K, prob.Ho * prob.Wo * prob.G * prob.K, 1, prob.Wo * prob.G * prob.K, prob.G * prob.K};
    std::array<std::size_t, 5> wei_strides{
        prob.K * prob.Y * prob.X * prob.C, prob.Y * prob.X * prob.C, 1, prob.X * prob.C, prob.C};
    std::array<std::size_t, 5> d_strides = {};

    std::vector<std::size_t> conv_filter_strides   = {2, 2};
    std::vector<std::size_t> conv_filter_dilations = {1, 1};
    std::vector<std::size_t> input_left_pads       = {1, 1};
    std::vector<std::size_t> input_right_pads      = {1, 1};
    static constexpr auto ConvSpec =
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
    static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
    // auto argument_ptr    = ck_args.MakeArgPtr(sh_conv_ptr, data_ctx.tensors); //FIXME: arg ptr
    // call -> use this? how is it passed in?

    for(auto solution : prob.GetSolutions("gfx908", prologue, epilogue))
    {
        auto src = ck::host::InterpolateString(
            conv_compile_check,
            {{"include", prob.GetIncludeHeader()},
             {"template", solution.ToTemplateString()},
             {"m", std::to_string(prob.G)},
             {"n", std::to_string(prob.N)},
             {"k", std::to_string(prob.C)}}); // FIXME: pass in the right dims
                                              
        std::ofstream ofh("kernel.txt");
        ofh << "##########################################################\n";
        ofh << solution.ToTemplateString();
        ofh << "##########################################################\n";
        ofh.close();

        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        options.kernel_name = "f";
        auto k              = rtc::compile_kernel(srcs, options);
        auto block_size     = solution.GetTemplateParameter<std::size_t>("BlockSize");
        auto m_per_block    = solution.GetTemplateParameter<ck::index_t>("MPerBlock");
        auto n_per_block    = solution.GetTemplateParameter<ck::index_t>("NPerBlock");
        auto k_per_block    = solution.GetTemplateParameter<ck::index_t>("KPerBlock");
        auto a_layout       = solution.GetTemplateParameter<std::string>("ALayout");
        auto grid_size      = ck::host::integer_divide_ceil(prob.G, m_per_block) *
                         ck::host::integer_divide_ceil(prob.N, n_per_block); // FIXME

	//auto A_Layout = ToLayout(a_layout);
        auto conv_to_gemm_transformer =
            ck::tensor_operation::TransformConvFwdToGemm<2, ConvSpec>{};

        auto matrix_padder = ck::tensor_operation::device::
            MatrixPadder<GemmSpec, ck::index_t, ck::index_t, ck::index_t>{
                m_per_block, n_per_block, k_per_block};

        /**const auto in_gemmmraw_gemmkraw_desc =
            conv_to_gemm_transformer.template MakeADescriptor_M_K<ALayout>(in_lengths,
                                                                        in_strides,
                                                                        wei_lengths,
                                                                        wei_strides,
                                                                        out_lengths,
                                                                        out_strides,
                                                                        conv_filter_strides,
                                                                        conv_filter_dilations,
                                                                        input_left_pads,
                                                                        input_right_pads);**/
        // creation of grid desc for run fcn here
        /**static constexpr auto GemmSpec = ck::host::GemmSpecialization::Default;
        // ck::host::GemmSpecialization GemmSpec = gemm_spec;
        static auto matrix_padder =
            ck::host::GemmPadder<GemmSpec, std::size_t, std::size_t, std::size_t>{
                m_per_block, n_per_block, k_per_block};
        auto NDimSpatial = prob.NumDim;
        // auto a_type  = solution.GetTemplateParameter<ck::host::DataType>("ADataType");
        const std::size_t tmp = prob.DsDataType.size();
        std::cout << tmp << std::endl;
        static constexpr auto NumDTensor = 0;
        static constexpr auto I1         = ck::host::Number<1>{};
        // input tensor desc
        const std::size_t N_in = in_lengths[1];
        const std::size_t C_in = in_lengths[2];

        const std::size_t Hi = in_lengths[3];
        const std::size_t Wi = in_lengths[4];

        const std::size_t Ho = out_lengths[3];
        const std::size_t Wo = out_lengths[4];

        const std::size_t ConvStrideH = conv_filter_strides[0];
        const std::size_t ConvStrideW = conv_filter_strides[1];
        const std::size_t Y           = wei_lengths[3];
        const std::size_t X           = wei_lengths[4];

        const std::size_t ConvDilationH = conv_filter_dilations[0];
        const std::size_t ConvDilationW = conv_filter_dilations[1];

        const std::size_t InLeftPadH = input_left_pads[0];
        const std::size_t InLeftPadW = input_left_pads[1];

        const std::size_t InRightPadH = input_right_pads[0];
        const std::size_t InRightPadW = input_right_pads[1];
        const std::size_t NStride     = in_strides[1];
        const std::size_t HiStride    = in_strides[3];
        const std::size_t WiStride    = in_strides[4];
        const auto CStride            = I1;

        const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
            ck::host::make_tuple(N_in, Hi, Wi, C_in),
            ck::host::make_tuple(NStride, HiStride, WiStride, CStride));

        const auto in_n_hip_wip_c_desc = ck::host::transform_tensor_descriptor(
            in_n_hi_wi_c_desc,
            ck::host::make_tuple(ck::host::make_pass_through_transform(N_in),
                                 ck::host::make_pad_transform(Hi, InLeftPadH, InRightPadH),
                                 ck::host::make_pad_transform(Wi, InLeftPadW, InRightPadW),
                                 ck::host::make_pass_through_transform(C_in)),
            ck::host::make_tuple(ck::host::Sequence<0>{},
                                 ck::host::Sequence<1>{},
                                 ck::host::Sequence<2>{},
                                 ck::host::Sequence<3>{}),
            ck::host::make_tuple(ck::host::Sequence<0>{},
                                 ck::host::Sequence<1>{},
                                 ck::host::Sequence<2>{},
                                 ck::host::Sequence<3>{}));

        const auto in_n_y_ho_x_wo_c_desc = ck::host::transform_tensor_descriptor(
            in_n_hip_wip_c_desc,
            ck::host::make_tuple(
                ck::host::make_pass_through_transform(N_in),
                ck::host::make_embed_transform(ck::host::make_tuple(Y, Ho),
                                               ck::host::make_tuple(ConvDilationH, ConvStrideH)),
                ck::host::make_embed_transform(ck::host::make_tuple(X, Wo),
                                               ck::host::make_tuple(ConvDilationW, ConvStrideW)),
                ck::host::make_pass_through_transform(C_in)),
            ck::host::make_tuple(ck::host::Sequence<0>{},
                                 ck::host::Sequence<1>{},
                                 ck::host::Sequence<2>{},
                                 ck::host::Sequence<3>{}),
            ck::host::make_tuple(ck::host::Sequence<0>{},
                                 ck::host::Sequence<1, 2>{},
                                 ck::host::Sequence<3, 4>{},
                                 ck::host::Sequence<5>{}));

        auto in_gemmm_gemmk_desc = ck::host::transform_tensor_descriptor(
            in_n_y_ho_x_wo_c_desc,
            ck::host::make_tuple(ck::host::make_merge_transform(ck::host::make_tuple(N_in, Ho, Wo)),
                                 ck::host::make_merge_transform(ck::host::make_tuple(Y, X, C_in))),
            ck::host::make_tuple(ck::host::Sequence<0, 2, 4>{}, ck::host::Sequence<1, 3, 5>{}),
            ck::host::make_tuple(ck::host::Sequence<0>{}, ck::host::Sequence<1>{}));
        using AGridDesc_M_K  = ck::host::remove_cvref_t<decltype(in_gemmm_gemmk_desc)>;
        // weight tensor desc
        const std::size_t K_wei = wei_lengths[1];
        const std::size_t C     = wei_lengths[2];

        const std::size_t YX = ck::host::accumulate_n<std::size_t>(
            wei_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

        const std::size_t KStride_wei = wei_strides[1];
        const std::size_t XStride     = wei_strides[2 + NDimSpatial];
        // const auto CStride           = I1; already declared

        const auto wei_k_yx_c_desc = ck::host::make_naive_tensor_descriptor(
            ck::host::make_tuple(K_wei, YX, C),
            ck::host::make_tuple(KStride_wei, XStride, CStride));

        auto wei_gemmn_gemmk_desc = ck::host::transform_tensor_descriptor(
            wei_k_yx_c_desc,
            ck::host::make_tuple(ck::host::make_pass_through_transform(K_wei),
                                 ck::host::make_merge_transform(ck::host::make_tuple(YX, C))),
            ck::host::make_tuple(ck::host::Sequence<0>{}, ck::host::Sequence<1, 2>{}),
            ck::host::make_tuple(ck::host::Sequence<0>{}, ck::host::Sequence<1>{}));

        // output tensor desc
        const std::size_t N     = out_lengths[1];
        const std::size_t K_out = out_lengths[2];

        const auto KStride_out     = I1;
        const std::size_t WoStride = out_strides[NDimSpatial + 2];

        const std::size_t NHoWo =
            N * ck::host::accumulate_n<std::size_t>( // FIXME: can i use CK methods??
                    out_lengths.begin() + 3,
                    NDimSpatial,
                    1,
                    std::multiplies<>());

        auto out_gemmm_gemmn_desc = ck::host::make_naive_tensor_descriptor(
            ck::host::make_tuple(NHoWo, K_out), ck::host::make_tuple(WoStride, KStride_out));
        // d desc
        // auto NumDTensor = prob.DsDataType
        auto d_grid_desc = ck::host::generate_tuple(
            [&](auto i) {
                // using DLayout = ck::host::remove_cvref_t<ck::host::tuple_element_t<i.value,
                // DsLayout>>;

                // using DLayout = ck::host::ToLayout(prob.DsTrans[0]);

                auto d_desc = ck::host::make_naive_tensor_descriptor(
                    ck::host::make_tuple(NHoWo, K_out),
                    d_strides[i]); // FIXME: get the right stride for Ds
            },
            ck::host::Number<NumDTensor>{});**/

        k.launch(nullptr, grid_size * block_size, block_size)(
            a.data(),
            b.data(),
            c.data(),
	    conv_to_gemm_transformer,
	    in_lengths,
	    in_strides, 
	    wei_lengths,
	    wei_strides,
	    out_lengths,
	    out_strides, 
	    conv_filter_strides,
	    conv_filter_dilations,
	    input_left_pads, 
	    input_right_pads
	    /**AGridDesc_M_K{},
            wei_gemmn_gemmk_desc,
            d_grid_desc,
            out_gemmm_gemmn_desc**/); // FIXME: my launch will bw different: will need
                                   // to pass in grid ptrs for run fcns
        CHECK(report(solution, check(rtc::from_gpu(c))));
    }
}
/**const std::string gemm_compile_check = R"__ck__(
#include <${include}>

extern "C" __global__ void f(const ck::half_t* a, const ck::half_t* b, ck::half_t* c) {
    using G = ${template};
    constexpr auto desc =
${template}::make_descriptor(ck::make_naive_tensor_descriptor_packed(ck::make_tuple(${m},
${k})), ck::make_naive_tensor_descriptor(ck::make_tuple(${n},
${k}), ck::make_tuple(1, ${n})), ck::make_tuple(),
                                             ck::make_naive_tensor_descriptor_packed(ck::make_tuple(${m},
${n})));

    static_assert(desc.IsValid(), "Invalid ck gemm.");

    if constexpr(desc.IsValid())
    {
        ${template}::Run(desc,
               a,
               b,
               ck::make_tuple(),
               c);
    }
}

)__ck__";

TEST_CASE(test_problem_kernel)
{
    ck::host::device_gemm_multiple_d::Problem prob;
    prob.M = 1024;
    prob.N = 1024;
    prob.K = 1024;
    check_all<half> check;
    auto a = to_gpu(generate_buffer<half>(1024 * 1024, 0));
    auto b = to_gpu(generate_buffer<half>(1024 * 1024, 1));
    auto c = to_gpu(generate_buffer<half>(1024 * 1024, 2));
    std::string prologue = "";
    std::string epilogue = "";

    for(auto solution : prob.GetSolutions("gfx908", prologue, epilogue))
    {
        auto src  = ck::host::InterpolateString(gemm_compile_check,
                                               {{"include", prob.GetIncludeHeader()},
                                                {"template", solution.ToTemplateString()},
                                                {"m", std::to_string(prob.M)},
                                                {"n", std::to_string(prob.N)},
                                                {"k", std::to_string(prob.K)}});
        auto srcs = get_headers_for_test();
        srcs.push_back({"main.cpp", src});
        rtc::compile_options options;
        options.kernel_name = "f";
        auto k              = rtc::compile_kernel(srcs, options);
        auto block_size     = solution.GetTemplateParameter<std::size_t>("BlockSize");
        auto m_per_block    = solution.GetTemplateParameter<std::size_t>("MPerBlock");
        auto n_per_block    = solution.GetTemplateParameter<std::size_t>("NPerBlock");
        auto grid_size      = ck::host::integer_divide_ceil(prob.M, m_per_block) *
                         ck::host::integer_divide_ceil(prob.N, n_per_block);
        k.launch(nullptr, grid_size * block_size, block_size)(a.data(), b.data(), c.data());
        CHECK(report(solution, check(rtc::from_gpu(c))));
    }
}**/

int main(int argc, const char* argv[]) { test::run(argc, argv); }
