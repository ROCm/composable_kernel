#include "ck_tile/host.hpp"
#include "reference_add_vector.hpp"
#include "add_vector.hpp"
#include <cstring>

// This example demonstrates how to use the ck_tile library to perform an elementwise vector
// addition using a custom kernel. The kernel is defined in the vector_add.hpp file, and the
// reference implementation is provided in the reference_vector_add.hpp file.

// parse command line arguments
// -m: size of the vectors
// -v: validation flag (1 for validation, 0 for no validation)
// -prec: precision of the data type (fp16, fp32, int8, int32)
// -warmup: number of warmup iterations (number of kernel launches before measuring performance)
// -repeat: number of repeat iterations (number of kernel launches to measure performance)
auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "256000000", "m dimension")
        .insert("v", "1", "cpu validation or not")
        .insert("prec", "fp16", "precision")
        .insert("warmup", "5", "cold iter")
        .insert("repeat", "20", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    using XDataType       = DataType; // input data type
    using ComputeDataType = float;    // compute data type
    using YDataType       = DataType; // output data type

    ck_tile::index_t m = arg_parser.get_int("m"); // size of the vectors
    int do_validation  = arg_parser.get_int("v"); // do we verify the result on cpu
    int warmup         = arg_parser.get_int("warmup");
    int repeat         = arg_parser.get_int("repeat");

    ck_tile::HostTensor<XDataType> x_host_a(
        {m}); // length input vector A, if given two arguments (m, n) the HostTensor will be created
              // with shape (m, n)
    ck_tile::HostTensor<XDataType> x_host_b(
        {m}); // length input vector B, if given two arguments (m, n) the HostTensor will be created
              // with shape (m, n)

    ck_tile::HostTensor<YDataType> y_host_ref({m});
    ck_tile::HostTensor<YDataType> y_host_dev({m});

    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(
        x_host_a); // fill the input vector A with random values
    ck_tile::FillUniformDistribution<XDataType>{-5.f, 5.f}(x_host_b);

    ck_tile::DeviceMem x_buf_a(
        x_host_a.get_element_space_size_in_bytes()); // allocate device memory for input vector A
                                                     // (this a wrapper over hipMalloc)
    ck_tile::DeviceMem x_buf_b(x_host_b.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y_host_dev.get_element_space_size_in_bytes());

    x_buf_a.ToDevice(
        x_host_a
            .data()); // copy the input vector A to device memory, this is a wrapper over hipMemcpy
    x_buf_b.ToDevice(x_host_b.data());

    // Dividing the problem into blocktile, warptile, and vector
    // The blocktile is the size of the tile that will be processed by a single thread block (also
    // called work group) The warptile is the size of the tile that will be processed by a single
    // warp (also called wavefront) The vector is the size of the tile that will be processed by a
    // single thread (also called work item) The problem is divided into blocks of size BlockTile,
    // each block is further divided into warps of size WarpTile and each warp is composed of 64 or
    // 32 threads of size Vector each of the thread in a warp will process one vector worth elements
    // of the data
    using BlockTile = ck_tile::sequence<8192>; // Size of the block tile (Entire problem is divided
                                               // into blocks of this size)
    using BlockWarps = ck_tile::sequence<8>; // How many concurrent warps are in a block (Each warp
                                             // will cover some part of blockTile)
    using WarpTile = ck_tile::sequence<64>;  // How many elements are covered by a warp
    using Vector   = ck_tile::sequence<1>; // How many elements are covered by a thread (Each thread
                                           // will cover some part of WarpTile)

    // Interpretation of above configurations
    // Each thread will cover 1 element (Vector)
    // Each WarpTile will cover 64 elements (WarpTile) --> since 64 threads in a warp
    // if we have 8 warps in a block (BlockWarps) then we have 8 * 64 = 512 threads in a block
    // if 8 warps are not enough to cover the entire blockTile then each of the 8 concurrent warps
    // will iterate over the blockTile several times

    constexpr ck_tile::index_t kBlockSize  = 512;
    constexpr ck_tile::index_t kBlockPerCu = 1;

    ck_tile::index_t kGridSize = (m / BlockTile::at(ck_tile::number<0>{}));
    std::cout << "block x-size = " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "grid size " << kGridSize << std::endl;

    using Shape = ck_tile::AddVectorShape<BlockWarps, BlockTile, WarpTile, Vector>;
    std::cout << "Problem Shape:: M = " << m << std::endl;
    std::cout << "BlockTile: " << BlockTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "Number of Blocks in Grid: " << m / BlockTile::at(ck_tile::number<0>{})
              << std::endl;
    std::cout << "BlockWarps: " << BlockWarps::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "WarpTile: " << WarpTile::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "Vector: " << Vector::at(ck_tile::number<0>{}) << std::endl;
    std::cout << "Repeat: " << Shape::Repeat_M
              << std::endl; // number of times a warp will iterate over the blockTile, covering
                            // different parts of the blockTile
    std::cout << "Threads per Block: " << kBlockSize << std::endl;
    std::cout << "ThreadBlocks per CU: " << kBlockPerCu << std::endl;

    // What is a Problem in CKTile?
    // A Problem defines the shape of the data, the precision of the data
    using Problem = ck_tile::AddVectorProblem<XDataType, ComputeDataType, YDataType, Shape>;

    // What is a Policy in CKTile?
    // A Policy defines how to map the data between threads and data in memory

    // The kernel is the function that will be executed on the device
    // It requires a Problem and Policy to be defined
    using Kernel = ck_tile::AddVectorKernel<Problem>;

    // The kernel is launched with the following parameters:
    float ave_time = launch_kernel(
        ck_tile::stream_config{nullptr, true, 0, warmup, repeat}, // wrapper over hipStreamCreate
        ck_tile::make_kernel<kBlockSize, kBlockPerCu>( // numOfThreadsPerBlock, numOfBlocksPerCU
            Kernel{},                                  // kernel
            kGridSize,                                 // number of blocks in the grid
            kBlockSize,                                // number of threads in a block
            0,                                         // shared memory size
            static_cast<XDataType*>(x_buf_a.GetDeviceBuffer()), // input vector A
            static_cast<XDataType*>(x_buf_b.GetDeviceBuffer()), // input vector B
            static_cast<YDataType*>(y_buf.GetDeviceBuffer()),   // output vector
            m));

    std::size_t num_btype = sizeof(XDataType) * m + sizeof(YDataType) * m;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    bool pass = true;

    if(do_validation)
    {
        ck_tile::reference_add_vector<XDataType, YDataType>(x_host_a, x_host_b, y_host_ref);
        y_buf.FromDevice(y_host_dev.mData.data());
        pass = ck_tile::check_err(y_host_dev, y_host_ref);

        std::cout << "valid:" << (pass ? "y" : "n") << std::flush << std::endl;
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
}
