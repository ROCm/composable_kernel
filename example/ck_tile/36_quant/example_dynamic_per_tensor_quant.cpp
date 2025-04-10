#include "ck_tile/host.hpp"

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/quant.hpp"
#include <cstring>
#include "ck_tile/host/reference/reference_rowwise_quantization2d.hpp"

// different threshold for different dtype
template <typename DataType>
auto get_elimit()
{
    double rtol = 1e-5;
    double atol = 1e-5;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    double rtol = 1e-5;
    double atol = 1e-5;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::int8_t>()
{
    // due to rounding, int8 quantization might have 1 abs error
    double rtol = 1;
    double atol = 1;
    return ck_tile::make_tuple(rtol, atol);
}


int main()
{

    static constexpr ck_tile::index_t Repeat_M_ = 8;
    static constexpr ck_tile::index_t Repeat_N_ = 1;
    
    static constexpr ck_tile::index_t ThreadPerBlock_M_ = 8;
    static constexpr ck_tile::index_t ThreadPerBlock_N_ = 64;

    static constexpr ck_tile::index_t Vector_N_ = 1;
    
    static constexpr bool is_warp_per_row = ThreadPerBlock_N_ <= warpSize;
    static_assert((ThreadPerBlock_M_ * ThreadPerBlock_N_) % warpSize == 0);
    static constexpr ck_tile::index_t total_warps =
        (ThreadPerBlock_M_ * ThreadPerBlock_N_) / warpSize;
    // std::cout<<"total_warps: "<<total_warps<<std::endl;
    // num of warps along m
    static constexpr ck_tile::index_t BlockWarps_M = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return total_warps * (warpSize / ThreadPerBlock_N_);
        }
        else
        {
            // static_assert(warpSize % ThreadPerBlock_M_ == 0);
            return total_warps / (ThreadPerBlock_N_ / warpSize);
        }
    }();

    // num of warps along n
    static constexpr ck_tile::index_t BlockWarps_N = []() {
        if constexpr(is_warp_per_row)
        {
            static_assert(warpSize % ThreadPerBlock_N_ == 0);
            return 1;
        }
        else
        {
            static_assert(ThreadPerBlock_N_ % warpSize == 0);
            return ThreadPerBlock_N_ / warpSize;
        }
    }();

    static constexpr ck_tile::index_t Block_M = Repeat_M_ * ThreadPerBlock_M_;
    static constexpr ck_tile::index_t Block_N = Repeat_N_ * ThreadPerBlock_N_ * Vector_N_;

    static constexpr ck_tile::index_t Warp_M = ThreadPerBlock_M_ / BlockWarps_M;
    static constexpr ck_tile::index_t Warp_N = ThreadPerBlock_N_ / BlockWarps_N * Vector_N_;

    using BlockTile  = ck_tile::sequence<Block_M, Block_N>;
    using BlockWarps = ck_tile::sequence<BlockWarps_M, BlockWarps_N>;
    using WarpTile   = ck_tile::sequence<Warp_M, Warp_N>;
    using Vector     = ck_tile::sequence<1, Vector_N_>;

    using Shape = ck_tile::Generic2dBlockShape<BlockTile, BlockWarps, WarpTile, Vector>;
    
    using XDataType = ck_tile::half_t;
    using ScaleDataType = float;
    using QXDataType = ck_tile::fp8_t;
    using ComputeDataType = float;

    using PipelineProblem = ck_tile::PerTensorQuantPipelineProblem<
        XDataType,
        ScaleDataType,
        ComputeDataType,
        QXDataType,
        Shape,
        true>;

    using Pipeline = ck_tile::DynamicPerTensorQuantPipeline<PipelineProblem>;

    using Kernel = ck_tile::PerTensorQuant<Pipeline>;

    int m = 64;
    int n = 64;
    int x_stride = 64;
    ck_tile::HostTensor<XDataType> x_host({m, n}, {x_stride, 1});
    ck_tile::HostTensor<ScaleDataType> scale_host({1}, {1});

    ck_tile::HostTensor<QXDataType> qx_host_ref({m, n}, {x_stride, 1});
    ck_tile::HostTensor<QXDataType> qx_host_dev({m, n}, {x_stride, 1});

    ck_tile::FillUniformDistribution<XDataType>{-.5f, .5f}(x_host);

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem scale_buf(scale_host.get_element_space_size_in_bytes());
    ck_tile::DeviceMem qx_buf(qx_host_dev.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    constexpr ck_tile::index_t kBlockPerCu = 1;
    Kernel::Kargs a{x_buf.GetDeviceBuffer(),
                    scale_buf.GetDeviceBuffer(),
                    qx_buf.GetDeviceBuffer(),
                    m,
                    n,
                    x_stride};
    
    const dim3 grids                       = Kernel::GridSize(a);
    constexpr dim3 blocks                  = Kernel::BlockSize();
    auto kargs = Kernel::MakeKargs(a);

    auto s = ck_tile::stream_config{nullptr, true, 1, 5, 10};
    auto time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    std::cout<<"time: "<<time<<std::endl;
    bool pass = true;

    scale_buf.FromDevice(scale_host.data());
    ck_tile::reference_per_tensor_quantization2d<XDataType, ScaleDataType, QXDataType>(
        x_host, scale_host, qx_host_ref);

    qx_buf.FromDevice(qx_host_dev.data());
    
    auto [rtol, atol] = get_elimit<QXDataType>();
    pass = ck_tile::check_err(qx_host_dev,
                              qx_host_ref,
                              std::string("qx Error: Incorrect results!"),
                              rtol,
                              atol);

    std::cout << (pass ? "pass" : "fail") << std::endl;
    return 0;
}
