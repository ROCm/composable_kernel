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

// 6)

#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_problem.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_traits.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_operators.hpp"
#include "ck_tile/ops/elementwise/pipeline/elementwise_pipeline_default_policy.hpp"
#include "ck_tile/ops/elementwise/kernel/elementwise.hpp"
#include "reference_add.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "10240", "m dimension")
        .insert("n", "4096", "n dimension")
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
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
    ck_tile::index_t stride = arg_parser.get_int("stride");
    if(stride < 0)
        stride = N;
    std::string data_type = arg_parser.get_str("prec");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    assert(stride >= N);

    using XDataType             = DataType;
    using ComputeDataType       = float;
    using YDataType             = DataType;
    using XElementwiseOperation = ck_tile::element_wise::Add;

    // 1. Initialize the input data on the host
    ck_tile::HostTensor<XDataType> x_host_a({M, N},
                                            {stride, 1}); // TODO: refactor to be more generic
    ck_tile::HostTensor<XDataType> x_host_b({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_validation({M, N}, {stride, 1});

    std::vector<ck_tile::index_t> shape = {M, N};
    ck_tile::index_t ndims = static_cast<ck_tile::index_t>(shape.size());

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
    using BlockWarps = ck_tile::sequence<16>;
    using BlockTile  = ck_tile::sequence<16384>;
    using WarpTile   = ck_tile::sequence<1024>;
    using Vector     = ck_tile::sequence<16>;
    
    using Shape   = ck_tile::ElementWiseTraits1D<BlockWarps, BlockTile, WarpTile, Vector>;
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;

    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy1D>;

    // Compute flattened size
    ck_tile::index_t MM = 1;
    for(int i = 0; i < ndims; ++i)
    {
        MM *= shape[i];
    }

    constexpr ck_tile::index_t kBlockSize  = 64 * BlockWarps::at(ck_tile::number<0>{});
    constexpr ck_tile::index_t kBlockPerCu = 1;
    constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
    ck_tile::index_t kGridSize = (MM + elements_per_block - 1) / elements_per_block;
    std::cout << "block x-size = " << elements_per_block << std::endl;
    std::cout << "grid size = " << kGridSize << std::endl;
    std::cout << "Total elements = " << MM << std::endl;
    

    // 4. Run the kernel
    float ave_time = launch_kernel(ck_tile::stream_config{nullptr, true, 0, warmup, repeat},
                  ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                      Kernel{},
                      kGridSize,
                      kBlockSize,
                      0,
                      static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()),
                      static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()),
                      static_cast<YDataType*>(y_buf.GetDeviceBuffer()),
                      MM));

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
