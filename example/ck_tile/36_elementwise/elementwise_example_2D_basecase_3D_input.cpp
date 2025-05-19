// TODO
// 1) Create main function
// 2) Create a function to initialize the input data
// 3) Create a function to run the kernel
// 4) Create verification function called `reference`
// 5) Create a function to verify the output

// TODO: Elementwise implementation
// 1) Pipeline
// 2) Policy
// 3) Kernel
// 4) Epilogue (?)
// 5) Implement a reference function (runs on the host, for verification)

#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_traits.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_operators.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp"
#include "ck_tile/ops/elementwise/kernel/elementwise.hpp"
#include "reference_add.hpp"

auto create_args(int argc, char* argv[])
{
    // 10240
    // 4096
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "4096", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("batch", "8", "batch size")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "0", "cold iter")
        .insert("repeat", "1", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t M      = arg_parser.get_int("m");
    ck_tile::index_t N      = arg_parser.get_int("n");
    ck_tile::index_t batch = arg_parser.get_int("batch");
    
    std::string data_type = arg_parser.get_str("prec");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    using XDataType             = DataType;
    using ComputeDataType       = float;
    using YDataType             = DataType;
    using XElementwiseOperation = ck_tile::element_wise::Add;

    auto lens = {batch, M, N};
    auto strides = {M*N, N, 1};

    // 1. Initialize the input data on the host
    ck_tile::HostTensor<XDataType> x_host_a(lens, strides); // TODO: refactor to be more generic
    ck_tile::HostTensor<XDataType> x_host_b(lens, strides);
    ck_tile::HostTensor<YDataType> y_host(lens, strides);
    ck_tile::HostTensor<YDataType> y_validation(lens, strides);

    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_b);

    // 2. Create device memory buffers
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes());
    x_buf_a.ToDevice(x_host_a.data());
    x_buf_b.ToDevice(x_host_b.data());
    y_buf.ToDevice(y_host.data());

    // 3. Create the kernel
    using BlockWarps = ck_tile::sequence<1, 8>;
    using BlockTile  = ck_tile::sequence<1, 4096>;
    using WarpTile   = ck_tile::sequence<1, 512>;
    using Vector     = ck_tile::sequence<1, 8>;

    constexpr ck_tile::index_t kBlockSize  = 512;
    constexpr ck_tile::index_t kBlockPerCu = 1;
    ck_tile::index_t kGridSize             = (M*batch / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "block x-size = " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape   = ck_tile::ElementWiseTraits2D<BlockWarps, BlockTile, WarpTile, Vector>;
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;

    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy2D>;

    auto input_tensors = ck_tile::make_tuple(static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()), 
                                            static_cast<XDataType*>(x_buf_b.GetDeviceBuffer())
                                        );
    // 4. Run the kernel
    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                  ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                      Kernel{},
                      kGridSize,
                      kBlockSize,
                      0,
                      static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                      ck_tile::make_tuple(batch, M, N),
                      ck_tile::make_tuple(M*N, N, 1)
                   ));
        
    std::cout << "Average time: " << ave_time << " ms" << std::endl;

    // 5. Verify the output
    bool pass = true;
    if(do_validation)
    {
        y_buf.FromDevice(y_validation.data());
        ck_tile::reference_add<XDataType, YDataType>(x_host_a, x_host_b, y_host);
        pass = ck_tile::check_err(
            y_validation, y_host, "Elementwise Add Error: Incorrect results!", 0.01, 0.01);
    }

    // std::cout<<std::endl<<y_host<<std::endl;
    // std::cout<<std::endl<<y_validation<<std::endl;

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
