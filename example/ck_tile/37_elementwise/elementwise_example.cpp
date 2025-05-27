#include "ck_tile/host.hpp"
#include "ck_tile/ops/elementwise.hpp"
#include "reference_add.hpp"

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "1024", "m dimension")
        .insert("n", "1024", "n dimension")
        .insert("stride", "-1", "stride per row, if -1 then equal to n")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "10", "cold iter")
        .insert("repeat", "50", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t M      = arg_parser.get_int("m");
    ck_tile::index_t N      = arg_parser.get_int("n");
    ck_tile::index_t stride = arg_parser.get_int("stride");

    // If stride is negative (default -1), set it to N, assuming a dense row-major layout.
    if(stride < 0)
        stride = N;
    std::string data_type = arg_parser.get_str("prec");
    int do_validation     = arg_parser.get_int("v");
    int warmup            = arg_parser.get_int("warmup");
    int repeat            = arg_parser.get_int("repeat");

    assert(stride >= N);

    // Define type aliases for clarity.
    // XDataType: Data type of the input tensors.
    // ComputeDataType: Data type used for intermediate computations (often float for precision).
    // YDataType: Data type of the output tensor.
    // XElementwiseOperation: The specific elementwise operation to perform (e.g., Add, Mul).
    using XDataType             = DataType;
    using ComputeDataType       = float; // Using float for intermediate calculations can improve numerical stability.
    using YDataType             = DataType;
    using XElementwiseOperation = ck_tile::element_wise::Add;

    // 1. Initialize the input data on the host (CPU).
    // HostTensor is a utility to manage tensor data on the CPU.
    // The first argument is the shape (dimensions) of the tensor {M, N}.
    // The second argument is the strides {stride, 1} for row-major layout.
    // 'x_host_a' and 'x_host_b' are the two input tensors for the elementwise operation.
    ck_tile::HostTensor<XDataType> x_host_a({M, N},
                                            {stride, 1});
    ck_tile::HostTensor<XDataType> x_host_b({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_host({M, N}, {stride, 1});
    ck_tile::HostTensor<YDataType> y_validation({M, N}, {stride, 1});

    std::vector<ck_tile::index_t> shape = {M, N};
    ck_tile::index_t ndims = static_cast<ck_tile::index_t>(shape.size());

    // Fill the host tensors with random data.
    // FillUniformDistribution populates the tensor with values from a uniform distribution,
    // within an interval.
    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_a);
    ck_tile::FillUniformDistribution<XDataType>{0.f, 5.f}(x_host_b);

    // 2. Create device memory buffers
    // DeviceMem allocates memory on the GPU.
    // The size is determined by the total number of elements and the size of DataType.
    ck_tile::DeviceMem x_buf_a(x_host_a.get_element_space_size_in_bytes());
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host.get_element_space_size_in_bytes());
    
    // Copy data from host tensors to device buffers.
    x_buf_a.ToDevice(x_host_a.data());
    x_buf_b.ToDevice(x_host_b.data());
    y_buf.ToDevice(y_host.data());

    // 3. Configure the kernel execution parameters.
    // Dividing the problem into blocktile, warptile, and vector
    // The blocktile is the size of the tile that will be processed by a single thread block (also
    // called work group) The warptile is the size of the tile that will be processed by a single
    // warp (also called wavefront) The vector is the size of the tile that will be processed by a
    // single thread (also called work item) The problem is divided into blocks of size BlockTile,
    // each block is further divided into warps of size WarpTile and each warp is composed of 64 or
    // 32 threads of size Vector each of the thread in a warp will process one vector worth elements
    // of the data
    // Note that WarpTile/ Vector should be a 64 for CDNA (because there 64 threads per warp)
    using BlockTile = ck_tile::sequence<2048>; // How many elements are handled by a block tile (the tensor is divided
                                               // into blocks of this size)
    using BlockWarps = ck_tile::sequence<8>; // How many concurrent warps are in a block (each warp
                                             // will cover some part of the block tile)

    // WarpTile: Defines the size of the data sub-tile processed by a single warp.
    // This should be consistent with BlockTile and BlockWarps.
    // If BlockTile is 2048 and BlockWarps is 8, then WarpTile could be 2048/8 = 256.
    // However, this example uses 64, meaning each warp processes 64 elements, and multiple
    // such warp operations might be needed to cover the BlockTile, or the BlockTile is
    // distributed differently.
    // The current configuration (BlockTile=2048, BlockWarps=8, WarpTile=64) implies that
    // each warp processes 64 elements, and 8 warps process 8*64 = 512 elements concurrently.                                             
    using WarpTile = ck_tile::sequence<64>;
    
    // Vector: Defines the number of elements processed by a single thread in one operation.
    // If Vector is sequence<1>, each thread handles one element at a time from its assigned WarpTile portion.
    // If WarpTile is 64 and warpSize is 64 (common), then each thread in the warp processes one element.
    // If Vector is > 1, it implies vectorized load/store/compute operations per thread.
    using Vector  = ck_tile::sequence<1>;


    // 4. Create the kernel

    // ElementWiseTraits bundles these tiling parameters.
    // It calculates derived properties like threads per warp, repeats, and total block size.
    using Shape   = ck_tile::ElementWiseTraits<BlockWarps, BlockTile, WarpTile, Vector>;

    // ElementWisePipelineProblem encapsulates all necessary information for the elementwise kernel:
    // - Data types (input, compute, output).
    // - Shape traits (tiling configuration).
    // - The specific elementwise operation (e.g., Add).
    using Problem = ck_tile::ElementWisePipelineProblem<XDataType,
                                                        ComputeDataType,
                                                        YDataType,
                                                        Shape,
                                                        XElementwiseOperation>;

    // ElementWiseKernel refers to the GPU kernel class
    using Kernel = ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

    // Compute flattened size
    ck_tile::index_t total_elements = 1;
    for(auto d : shape) total_elements *= d;
    
    
    // kBlockSize: The number of threads in a GPU thread block (workgroup).
    // This is often a multiple of the warp size, 64 on CDNA.
    // Here, it's explicitly set to 512. This should be consistent with Shape::kBlockSize.
    // Shape::kBlockSize would be BlockWarps * warpSize (e.g., 8 * 64 = 512).
    constexpr ck_tile::index_t kBlockSize  = 64 * BlockWarps::at(ck_tile::number<0>{});

    // kBlockPerCu: Hint for how many thread blocks can be scheduled per Compute Unit (CU).
    // This can influence occupancy and performance.
    constexpr ck_tile::index_t kBlockPerCu = 1;
    
    // kGridSize: Calculates the total number of thread blocks required to process all elements.
    // Each thread block is responsible for 'elements_per_block' elements.
    // To ensure all elements are covered, especially when 'total_elements' is not perfectly
    // divisible by 'elements_per_block', using ceiling division.
    constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
    ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;
    
    std::cout << "grid size = " << kGridSize << std::endl;
    std::cout << "Total elements = " << total_elements << std::endl;
    
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
                      ck_tile::make_tuple(M, N), // Input size
                      ck_tile::make_tuple(N, 1), // Stride
                      input_tensors,
                      static_cast<YDataType*>(y_buf.GetDeviceBuffer())
                    ));

    std::cout << "Average time: " << ave_time << " ms" << std::endl;
    
    // 5. Verify the output
    bool pass = true;
    if(do_validation)
    {
        y_buf.FromDevice(y_validation.data());
        ck_tile::reference_add<XDataType, YDataType>(y_host, x_host_a, x_host_b);
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
