// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

//#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"


using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

namespace ck {
namespace tensor_operation {
namespace device {
static constexpr index_t WaveSize = 64;
static constexpr index_t W_PACK  = 2; // WaveSize / arg.input_spatial_lengths_[1];
static constexpr index_t Tile_H   = 32;
static constexpr index_t Tile_W   = 32;
static constexpr index_t N_Pack   = 2;
static constexpr index_t Pad_H    = 2;
static constexpr index_t Pad_W    = 2;
static constexpr index_t Filter_X = 5;
static constexpr index_t Filter_Y = 5;

static constexpr index_t SizeOfType = 2;
static constexpr index_t Tile_Align_W = Tile_W;
static constexpr index_t ShareMemSize = Tile_H * Tile_Align_W * N_Pack * SizeOfType;
static constexpr index_t ScratchSize = ShareMemSize / 64 / 4;
static constexpr index_t Num_Wave = 4;
#define MergeShareMem 0
template <typename T>
__device__ T warp_shuffle_up(const T& v_local, uint32_t lane_delta)
{
#if 0
    return  __shfl_up(v_local, lane_delta);
#elif 1
    static_assert(sizeof(T) == sizeof(int32_t), "wrong!");

    const uint32_t wrap_around_lane_delta = warpSize - lane_delta;

    const int32_t v_remote_tmp = __builtin_amdgcn_ds_bpermute(
        (__lane_id() << 2) + (wrap_around_lane_delta << 2), bit_cast<int32_t>(v_local));

    return bit_cast<T>(v_remote_tmp);
#endif
}

template <typename T>
__device__ T warp_shuffle_down(const T& v_local, uint32_t lane_delta)
{
#if 0
    return  __shfl_down(v_local, lane_delta);
#elif 1
    static_assert(sizeof(T) == sizeof(int32_t), "wrong!");

    const int32_t v_remote_tmp = __builtin_amdgcn_ds_bpermute(
        (__lane_id() << 2) + (lane_delta << 2), bit_cast<int32_t>(v_local));

    return bit_cast<T>(v_remote_tmp);
#endif
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
template <typename DataType>
void __device__ load_data_from_global(DataType* p,
                                      index_t lane_id,
                                     index_t n_stride,
                                       index_t h,
                                       index_t w,
                                       index_t h_stride,
                                       index_t w_stride,
                                       uint32_t* p_scratch)
{
    ignore = h;
    ignore = w;
    DataType* p_1 = p + n_stride;
    static_assert(sizeof(DataType) == 2);
    static_assert(Pad_H % W_PACK == 0);

    const index_t x = lane_id % (WaveSize/W_PACK);
    const index_t y_base = lane_id / (WaveSize/W_PACK);

    auto get_offset = [&](index_t y_, index_t x_)
    {
        return y_ * h_stride + x_ * w_stride;
    };

    if(x >= Pad_W && x < w + Pad_W)
    {
        static_for<0, Tile_H / W_PACK, 1>{}([&](auto i) {
            const index_t y = y_base + i * W_PACK;
            if constexpr (i * W_PACK >= Pad_H && i * W_PACK < (Tile_H - Pad_H))
            {
                const index_t offset = get_offset(y - Pad_H, x - Pad_W);
                half2_t tmp          = {};
                tmp[0]               = p[offset];
                tmp[1]               = p_1[offset];
                p_scratch[i]         = bit_cast<uint32_t>(tmp);
            }
        });
    }
}

void __device__ write_data_to_lds(index_t lane_id, const uint32_t* p_scratch, uint32_t* p_sharemem)
{
    const index_t x = lane_id % (WaveSize/W_PACK);
    const index_t y_base = lane_id / (WaveSize/W_PACK);
    //static_assert(N_Pack * sizeof(InDataType)/ sizeof(uint32_t) == 1);

    auto get_offset = [&](index_t y_, index_t x_) { return (y_ * Tile_Align_W + (x_));}; // + y_ * Filter_X) % Tile_W); };
    static_for<0, Tile_H / W_PACK, 1>{}([&](auto i) {
        const index_t y      = y_base + i * W_PACK;
        const index_t offset = get_offset(y, x);
        p_sharemem[offset]   = p_scratch[i];
    });
}

void __device__ run_conv_bwd_weight(index_t x, index_t y, index_t H, index_t W, index_t H_base, uint32_t* p_share_in, uint32_t* p_share_out,float& acc)
{
    ignore = H;
    ignore = W;
    auto get_in = [&](int h_, int w_)
    {
        return p_share_in[(h_ + y + H_base) * Tile_Align_W + (w_ + x)];// + (h_ + y + H_base) * Filter_X) % Tile_W];
    };
    auto get_out = [&](int h_, int w_)
    {
        return p_share_out[(h_ + Pad_H + H_base) * Tile_Align_W + (w_ + Pad_W)];// + (h_ + Pad_H + H_base) * Filter_X) % Tile_W];
    };
    if (x < Filter_X && y < Filter_Y)
    {
        //for (int ho = 0; ho < H; ho++)
        static_for<0, (Tile_H - Pad_H - Pad_H) / 2, 1>{}([&](auto ho)
        {
            //for (int wo = 0; wo < W; wo++)
            static_for<0, Tile_W - Pad_W - Pad_W, 1>{}([&](auto wo)
            {
                uint32_t v_in = get_in(ho, wo);
                uint32_t v_out = get_out(ho, wo);
                acc = __builtin_amdgcn_fdot2(bit_cast<half2_t>(v_in), bit_cast<half2_t>(v_out), acc, false);
            });
        });
    }
}
template <typename Argument, typename WeiDataType>
void __device__ write_output(const Argument& arg, int g, int y, int x, WeiDataType acc)
{
    const index_t Wei_G_Stride = arg.wei_g_k_c_xs_strides_[0];
    const index_t Y_Stride     = arg.wei_g_k_c_xs_strides_[3];
    const index_t X_Stride     = arg.wei_g_k_c_xs_strides_[4];

    if (y < Filter_Y && x < Filter_X)
    {
        auto p_wei = arg.p_wei_grid_ + Wei_G_Stride * g + y * Y_Stride + x * X_Stride;
        *p_wei = acc;
    }
}


template <typename Argument, index_t MinimumOccupancy = 1>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(WaveSize * Num_Wave, MinimumOccupancy)
#endif
        kernel_grouped_conv_bwd_weight_naive(Argument arg)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const index_t wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WaveSize);
    const index_t lane_id = __lane_id();
    //index_t n_idx = 0;

#if MergeShareMem
    constexpr index_t ShaderMemSizePerWave =ShareMemSize * 2  - Pad_H * Tile_Align_W * sizeof(uint32_t);
    __shared__ uint32_t p_share_mem[ShaderMemSizePerWave/sizeof(uint32_t)  * Num_Wave];
    auto* p_share_in = &p_share_mem[0];
    //__shared__ char p_input_1[ShareMemSize];
    auto* p_share_out = &p_share_mem[(ShareMemSize - Pad_H * Tile_Align_W * sizeof(uint32_t))/ sizeof(uint32_t)];
#else
    __shared__ uint32_t p_share_in[ShareMemSize/sizeof(uint32_t)  * Num_Wave];
    //__shared__ char p_input_1[ShareMemSize];
    __shared__ uint32_t p_share_out[ShareMemSize/sizeof(uint32_t) * Num_Wave];
    //__shared__ char p_output_1[ShareMemSize];
#endif
    uint32_t p_input_0_scratch[ScratchSize];
    uint32_t p_input_1_scratch[ScratchSize];
    uint32_t p_output_0_scratch[ScratchSize];
    uint32_t p_output_1_scratch[ScratchSize];

    static constexpr index_t spatial_offset = 3;
    //const index_t G = arg.in_g_n_c_wis_lengths[0];
    const index_t N = arg.in_g_n_c_wis_lengths_[1];
    index_t num_loop = N / Num_Wave / 2 - 1;
    index_t n_idx = N / Num_Wave * wave_id;
    if (wave_id == Num_Wave - 1)
    {
        n_idx = N / Num_Wave * (Num_Wave - 1);
        num_loop = (N - n_idx)/2 -1;
    }

    // In
    const index_t Hi = arg.in_g_n_c_wis_lengths_[spatial_offset + 0];
    const index_t Wi = arg.in_g_n_c_wis_lengths_[spatial_offset + 1];

    const index_t Hi_Stride   = arg.in_g_n_c_wis_strides_[spatial_offset + 0];
    const index_t Wi_Stride   = arg.in_g_n_c_wis_strides_[spatial_offset + 1];
    const index_t In_G_Stride = arg.in_g_n_c_wis_strides_[0];
    const index_t In_N_Stride = arg.in_g_n_c_wis_strides_[1];

    // Out
    const index_t Ho = arg.out_g_n_k_wos_lengths_[spatial_offset + 0];
    const index_t Wo = arg.out_g_n_k_wos_lengths_[spatial_offset + 1];

    const index_t Ho_Stride    = arg.out_g_n_k_wos_strides_[spatial_offset + 0];
    const index_t Wo_Stride    = arg.out_g_n_k_wos_strides_[spatial_offset + 1];
    const index_t Out_G_Stride = arg.out_g_n_k_wos_strides_[0];
    const index_t Out_N_Stride = arg.out_g_n_k_wos_strides_[1];

    static_for<0,ScratchSize, 1>{}([&](auto i)
    {
        p_input_0_scratch[i] = 0;
        p_output_0_scratch[i] = 0;
        p_input_1_scratch[i] = 0;
        p_output_1_scratch[i] = 0;
    });
    // Wei

    //static_assert(sizeof(InDataType) == 2);
    //static_assert(sizeof(OutputDataType) == 2);
    //
    auto* p_in   = arg.p_in_grid_ + g_idx * In_G_Stride + n_idx * In_N_Stride;
    auto* p_out = arg.p_out_grid_ + g_idx * Out_G_Stride + n_idx * Out_N_Stride;

    // prefetch 0
    load_data_from_global(p_in, lane_id, In_N_Stride, Hi, Wi, Hi_Stride, Wi_Stride, p_input_0_scratch);
    load_data_from_global(p_out, lane_id, Out_N_Stride, Ho, Wo, Ho_Stride, Wo_Stride, p_output_0_scratch);
    //load_data_from_global(p_in + In_N_Stride,  Hi, Wi, Hi_Stride, Wi_Stride, p_input_1_scratch);
    //load_data_from_global(p_out + Out_N_Stride, Ho, Wo, Ho_Stride, Wo_Stride, p_output_1_scratch);
    p_in += 2 * In_N_Stride;
    p_out += 2 * Out_N_Stride;

    // prefetch 1
    //load_input_from_global(arg, p_in, n_idx + 1, p_input_1_scratch);
    //load_output_from_global(arg, p_out, n_idx + 1, p_output_0_scratch);

    // write 0
    auto p_input_0 = p_share_in + ShareMemSize/sizeof(uint32_t)  * wave_id;
    auto p_output_0 = p_share_out + ShareMemSize/sizeof(uint32_t)  * wave_id;

    write_data_to_lds( lane_id,p_input_0_scratch, p_input_0);
    write_data_to_lds( lane_id,p_output_0_scratch, p_output_0);
    index_t H_base = lane_id >=32 ? (Tile_H - Pad_H - Pad_H) /2 : 0;
    index_t x = (lane_id % 32) % Filter_X;
    index_t y = (lane_id % 32) / Filter_X;
    float acc = 0;


    while(num_loop > 0)
    {
        // prefetch 0
        load_data_from_global(p_in,   lane_id,In_N_Stride, Hi, Wi, Hi_Stride, Wi_Stride, p_input_0_scratch);
        load_data_from_global(p_out, lane_id, Out_N_Stride, Ho, Wo, Ho_Stride, Wo_Stride, p_output_0_scratch);
        //load_data_from_global(p_in + In_N_Stride,  Hi, Wi, Hi_Stride, Wi_Stride, p_input_1_scratch);
        //load_data_from_global(p_out + Out_N_Stride,  Ho, Wo, Ho_Stride, Wo_Stride, p_output_1_scratch);
        p_in += 2 * In_N_Stride;
        p_out += 2 * Out_N_Stride;
 
        // do conv_bwd on 0
            block_sync_lds();
        run_conv_bwd_weight(x, y, Ho, Wo, H_base, p_input_0, p_output_0, acc);
    block_sync_lds();
        // write 1
        //write_input_to_lds();
        //write_output_to_lds();

        // prefetch 1
        //load_input_from_global();
        //load_output_from_global();
        // do conv_bwd on 1
        //run_conv_bwd_weight();

        // write 0
        write_data_to_lds( lane_id,p_input_0_scratch, p_input_0);
        write_data_to_lds( lane_id,p_output_0_scratch, p_output_0);

        num_loop --;
    };

    // tail
    {
        run_conv_bwd_weight(x, y, Ho, Wo, H_base, p_input_0, p_output_0, acc);
    }
    float acc_2 = warp_shuffle_down(acc, 32);
    acc += acc_2;
    block_sync_lds();
    p_share_in[threadIdx.x] = bit_cast<uint32_t>(acc);
    block_sync_lds();
    if (H_base == 0 && wave_id == 0)
    { 
        for(int i= 1; i < Num_Wave; i++)
        {
            acc += bit_cast<float>(p_share_in[i * WaveSize + lane_id]); 
        }
        write_output(arg, g_idx, y, x, acc);
    }

#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}
#pragma clang diagnostic pop

template <ck::index_t NDimSpatial,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA
          >
struct DeviceGroupedConvBwdWeightNaive
    : public DeviceGroupedConvBwdWeight<NDimSpatial,
                                        InLayout,
                                        WeiLayout,
                                        OutLayout,
                                        InDataType,
                                        WeiDataType,
                                        OutDataType,
                                        InElementwiseOperation,
                                        WeiElementwiseOperation,
                                        OutElementwiseOperation,
                                        ComputeTypeA,
                                        ComputeTypeB>
{
    using DeviceOp = DeviceGroupedConvBwdWeightNaive;
    static_assert(is_same_v<InElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<WeiElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<OutElementwiseOperation, element_wise::PassThrough>);

    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
            : p_in_grid_{p_in_grid},
              p_wei_grid_{p_wei_grid},
              p_out_grid_{p_out_grid},
              out_element_op_{out_element_op},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              in_g_n_c_wis_lengths_(in_g_n_c_wis_lengths),
              in_g_n_c_wis_strides_(in_g_n_c_wis_strides),
              wei_g_k_c_xs_lengths_(wei_g_k_c_xs_lengths),
              wei_g_k_c_xs_strides_(wei_g_k_c_xs_strides),
              out_g_n_k_wos_lengths_(out_g_n_k_wos_lengths),
              out_g_n_k_wos_strides_(out_g_n_k_wos_strides),
              conv_filter_strides_(conv_filter_strides),
              conv_filter_dilations_(conv_filter_dilations),
              input_left_pads_(input_left_pads),
              input_right_pads_(input_right_pads),
              k_batch_{split_k}
        {
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            return 0;
        }

        const InDataType* p_in_grid_;
        WeiDataType* p_wei_grid_;
        const OutDataType* p_out_grid_;

        OutElementwiseOperation out_element_op_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;

        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_lengths_; 
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_strides_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_strides_;
        std::array<ck::index_t, NDimSpatial> conv_filter_strides_;
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations_;
        std::array<ck::index_t, NDimSpatial> input_left_pads_;
        std::array<ck::index_t, NDimSpatial> input_right_pads_;
        const index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument&)
        {

        }
        index_t CalculateGridSize(const Argument& arg)
        {
            return arg.in_g_n_c_wis_lengths_[0];;
        }

        float RunGemmV3(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
    
            index_t gdx = CalculateGridSize(arg);

            float ave_time = 0;

           constexpr index_t minimum_occupancy = 1;
           //     BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;
            constexpr index_t BlockSize = Num_Wave * WaveSize;
            const auto kernel = kernel_grouped_conv_bwd_weight_naive<
                Argument, minimum_occupancy>;

            ave_time += launch_and_time_kernel(
                        stream_config, 
                        kernel,
                        dim3(gdx),
                        dim3(BlockSize),
                        0,
                        arg);

            return ave_time;
        }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {         
            float avg_time                 = 0.f;
            avg_time += RunGemmV3(arg, stream_config);
            return avg_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument&)
    {
        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 const ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        b_g_n_c_wis_lengths, // input
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths, // weight
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths, // output
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        void* p_wei_grid,
                        const void* p_out_grid,
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        const ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          b_g_n_c_wis_lengths, // input
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths, // weight
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths, // output
                                          a_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        return "";
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = dynamic_cast<const Argument*>(p_arg);
        if(arg)
        {
            return arg->GetWorkspaceSizeBytes();
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument structure!");
    }
};


}
}
}

using ALayout = ck::tensor_layout::convolution::GNHWC;
using BLayout = ck::tensor_layout::convolution::GKYXC;
using ELayout = ck::tensor_layout::convolution::GNHWK;

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightNaive<NDimSpatial,
                                                                  ALayout,
                                                                  BLayout,
                                                                  ELayout,
                                                                  InDataType,
                                                                  WeiDataType,
                                                                  OutDataType,
                                                                  InElementOp,
                                                                  WeiElementOp,
                                                                  OutElementOp>;

template <ck::index_t NDimSpatial>
using HostConvBwdWeightInstance = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                                     InDataType,
                                                                                     WeiDataType,
                                                                                     OutDataType,
                                                                                     InElementOp,
                                                                                     WeiElementOp,
                                                                                     OutElementOp>;

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return 1;
    }

    switch(conv_param.num_dim_spatial_)
    {
    case 1: break;//return !run_grouped_conv_bwd_weight<1>(config, conv_param);
    case 2: return !run_grouped_conv_bwd_weight<2>(config, conv_param);
    case 3: break;//return !run_grouped_conv_bwd_weight<3>(config, conv_param);
    default: break;
    }

    return 1;
}
