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

#if 0
template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeight_Xdl_CShuffle<
        NDimSpatial,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GNWC,
                                      ck::tensor_layout::convolution::GNHWC,
                                      ck::tensor_layout::convolution::GNDHWC>>,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GKXC,
                                      ck::tensor_layout::convolution::GKYXC,
                                      ck::tensor_layout::convolution::GKZYXC>>,
        ck::tuple_element_t<NDimSpatial - 1,
                            ck::Tuple<ck::tensor_layout::convolution::GNWK,
                                      ck::tensor_layout::convolution::GNHWK,
                                      ck::tensor_layout::convolution::GNDHWK>>,
        InDataType,           // InDataType
        WeiDataType,          // WeiDataType
        OutDataType,          // OutDataType
        AccDataType,          // AccDataType
        InElementOp,          // InElementwiseOperation
        WeiElementOp,         // WeiElementwiseOperation
        OutElementOp,         // OutElementwiseOperation
        ConvBwdWeightDefault, // ConvolutionBackwardWeightSpecialization
        256,                  // BlockSize
        128,                  // MPerBlock
        128,                  // NPerBlock
        4,                    // K0PerBlock
        8,                    // K1
        32,                   // MPerXdl
        32,                   // NPerXdl
        2,                    // MXdlPerWave
        2,                    // NXdlPerWave
        S<1, 4, 16, 4>,       // ABlockTransferThreadClusterLengths_K0_M_K1
        S<0, 3, 1, 2>,        // ABlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,        // ABlockTransferSrcAccessOrder
        2,                    // ABlockTransferSrcVectorDim
        1,                    // ABlockTransferSrcScalarPerVector
        1,                    // ABlockTransferDstScalarPerVector_K1
        true,                 // ABlockLdsAddExtraM
        S<1, 4, 16, 4>,       // BBlockTransferThreadClusterLengths_K0_N_K1
        S<0, 3, 1, 2>,        // BBlockTransferThreadClusterArrangeOrder
        S<0, 2, 1, 3>,        // BBlockTransferSrcAccessOrder
        2,                    // BBlockTransferSrcVectorDim
        1,                    // BBlockTransferSrcScalarPerVector
        1,                    // BBlockTransferDstScalarPerVector_K1
        true,                 // BBlockLdsAddExtraN
        1,                    // CShuffleMXdlPerWavePerShuffle
        1,                    // CShuffleNXdlPerWavePerShuffle
        S<1, 32, 1, 4>,       // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        1>; // CBlockTransferScalarPerVector_NWaveNPerXdl

#endif

//       DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,    32,     32,   8,   32,   32,    1,    1,  S<4, 8,  1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              2,              2,      false,  S<4, 16,  1>, S<2, 0, 1>,  S<1, 0, 2>,                1,              2,              2,      false,           1,           1,   S<1, 8, 1, 8>,                  1, Scheduler, PipelineVersion, 2>,

 //ConvBwdWeightDefault,
//is_NHWGC_GKYXC_NHWGK
using ALayout  = ck::tensor_layout::convolution::NHWGC;
using BLayout = ck::tensor_layout::convolution::GKYXC;
using ELayout = ck::tensor_layout::convolution::NHWGK;
//using Scheduler =ck::BlockGemmPipelineScheduler::Intrawave;
//using PipelineVersion =ck::BlockGemmPipelineVersion::v1;
template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
    ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<
        NDimSpatial,
        ALayout,
        BLayout,
        ELayout,
        F16,
        F16,
        F16,
        F32,
        PassThrough,
        PassThrough,
        PassThrough,
        ConvBwdWeightDefault,
        64,
        32,//16,
        64,
        32,//64,
        8,
        32, //16,
        32, //16,
        1,
        2, //4,
        S<4, 8, 1>,// S<8, 8, 1>
        S<2, 0, 1>,
        S<1, 0, 2>,
        1,
        2,
        2,
        false,
        S<4, 16, 1>,
        S<2, 0, 1>,
        S<1, 0, 2>,
        1,
        2,
        2,
        false,
        1,
        1,
        S<1, 8, 1, 8>,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        2>;
#if 0

       DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle< NDimSpatial,  ALayout,   BLayout,   ELayout,    F16,     F16,     F16,     F32, PassThrough, PassThrough, PassThrough,                  ConvSpec,    64,    32,    64,     32,   8,   32,   32,    1,    2,  S<4, 8,  1>, S<2, 0, 1>,  S<1, 0, 2>,                   1,              2,              2,      false,  S<4, 16,  1>, S<2, 0, 1>,  S<1, 0, 2>,                1,              2,              2,      false,           1,           1,   S<1, 8, 1, 8>,                  1, Scheduler, PipelineVersion, 2>,


64, 16, 16, 32, 8, 16, 16, 1, 1, S<4, 8, 1>, S<2, 0, 1>, S<1, 0, 2>, 1, 1, 4, false, S<4, 8, 1>,
    S<2, 0, 1>, S<1, 0, 2>, 1, 1, 4, false, 1, 1, S<1, 8, 1, 8>, 1,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, 8 > ;
#endif

namespace ck {
namespace tensor_operation {
namespace device {

static constexpr index_t WaveSize = 64;
template<typename Argument,
         typename InDataType>
void __device__ load_input_from_global(const Argument* arg, InDataType* p_in, index_t n, uint32_t* p_scratch)
{
    InDataType* p_in_n = p_in + arg->a_g_n_k_wos_strides[1] * n;
    InDataType* p_in_n_1 = p_in + arg->a_g_n_k_wos_strides[1] * (n + 1);

    const uint32_t W_PACK = 2; //WaveSize / arg->input_spatial_lengths_[1];
    static_assert(sizeof(InDataType) == 2);

    auto get_offset = [&](index_t y, index_t x)
    {
        return y * arg->input_spatial_stride_[0] + x * arg->input_spatial_stride_[1];
    }
    for (uint32_t i = 0; i < arg->input_spatial_lengths_[1] / W_PACK; i++)
    {
        const index_t offset = get_offset(i * W_PACK + threadIdx.x / (64/W_PACK), threadIdx.x % (64/W_PACK));
        auto tmp0 = p_in_n[offset];
        auto tmp1 = p_in_n_1[offset];
        InDataType* p_scratch_offset = reinterpret_cast<InDataType*>(&p_scratch[i]);
        p_scratch_offset[0] = tmp1;
        p_scratch_offset[1] = tmp1; 
    }
}

template<typename Argument,
         typename OutDataType>
void __device__ load_output_from_global(const Argument* arg, OutDataType* p_out, index_t n, uint32_t* p_scratch)
{
    OutDataType* p_out_n = p_out + arg->a_g_n_k_wos_strides[1] * n;
    OutDataType* p_out_n_1 = p_out + arg->a_g_n_k_wos_strides[1] * (n + 1);

    const uint32_t W_PACK = 2; //WaveSize / arg->input_spatial_lengths_[1];
    static_assert(sizeof(OutDataType) == 2);

    auto get_offset = [&](index_t y, index_t x)
    {
        return y * arg->output_spatial_stride_[0] + x * arg->output_spatial_stride_[1];
    }
    for (uint32_t i = 0; i < arg->output_spatial_lengths_[1] / W_PACK; i++)
    {
        const index_t offset = get_offset(i * W_PACK + threadIdx.x / (64/W_PACK), threadIdx.x % (64/W_PACK));
        auto tmp0 = p_out_n[offset];
        auto tmp1 = p_out_n_1[offset];
        InDataType* p_scratch_offset = reinterpret_cast<InDataType*>(&p_scratch[i]);
        p_scratch_offset[0] = tmp1;
        p_scratch_offset[1] = tmp1; 
    }
}

write_input_to_lds

template <typename Argument, index_t MinimumOccupancy = 1>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, MinimumOccupancy)
#endif
        kernel_grouped_conv_bwd_weight_naive(Argument* arg)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx9__))
    const index_t g_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    index_t n_idx = 0;

    constexpr index_t Tile_H = 32;
    constexpr index_t Tile_W = 32;
    constexpr index_t N_Pack = 2;
    constexpr index_t SizeOfType = 2;
    constexpr index_t ShareMemSize = Tile_H * Tile_W * N_Pack * SizeOfType;
    __shared__ char p_input_0[ShareMemSize];
    __shared__ char p_input_1[ShareMemSize];
    __shared__ char p_output_0[ShareMemSize];
    __shared__ char p_output_1[ShareMemSize];

    constexpr index_t ScratchSize = ShareMemSize / 64 / 4;
    uint32_t p_input_0_scratch[ScratchSize];
    uint32_t p_input_1_scratch[ScratchSize];
    uint32_t p_output_0_scratch[ScratchSize];
    uint32_t p_output_1_scratch[ScratchSize];

    InDataType* p_in = arg->p_in_grid + g_idx * arg->a_g_n_k_wos_strides[0];
    OutDataType* p_out = arg->p_out_grid + g_idx * arg->e_g_k_c_xs_strides[0];

    // prefetch 0
    load_input_from_global(arg, p_in, n_idx, p_input_0_scratch);
    load_output_from_global(arg, p_out, n_idx, p_output_0_scratch);
    // prefetch 1
    load_input_from_global(arg, p_in, n_idx + 1, p_input_1_scratch);
    load_output_from_global(arg, p_out, n_idx + 1, p_output_0_scratch);

    // write 0
    write_input_to_lds(arg, p_input_0_scratch);
    write_output_to_lds(arg, p_output_0_scratch);

    while(num_loop > 0)
    {
        // prefetch 0
        load_input_from_global();
        load_output_from_global();
        // do conv_bwd on 0
        run_conv_bwd_weight();

        // write 1
        write_input_to_lds();
        write_output_to_lds();

        // prefetch 1
        load_input_from_global();
        load_output_from_global();
        // do conv_bwd on 1
        run_conv_bwd_weight();

        // write 0
        write_input_to_lds();
        write_output_to_lds();

        num_loop --;
    };

    if (tail_num == 1)
    {

    }

    if (tail_num == 2)
    {

    }

    write_output();

#else
    ignore = karg;
#endif // end of if (defined(__gfx9__))
}

template <ck::index_t NDimSpatial
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
    static_assert(is_same_v<InElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<WeiElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<OutElementwiseOperation, element_wise::PassThrough>);

    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
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
                 ck::index_t split_k)
            : p_a_grid_{p_out_grid},
              p_b_grid_{p_in_grid},
              p_e_grid_{p_wei_grid},
              a_grid_desc_k0_m_k1_{},
              b_grid_desc_k0_n_k1_{},
              ce_grid_desc_m_n_{},
              c_grid_desc_mblock_mperblock_nblock_nperblock_{},
              compute_ptr_offset_of_batch_{},
              a_element_op_{out_element_op},
              b_element_op_{in_element_op},
              cde_element_op_{wei_element_op},
              Conv_G_{b_g_n_c_wis_lengths[0]},
              Conv_N_{b_g_n_c_wis_lengths[1]},
              Conv_K_{e_g_k_c_xs_lengths[1]},
              Conv_C_{b_g_n_c_wis_lengths[2]},
              input_spatial_lengths_{},
              filter_spatial_lengths_{},
              output_spatial_lengths_{},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads},
              k_batch_{split_k}
        {
            constexpr index_t spatial_offset = 3;
            std::copy(begin(b_g_n_c_wis_lengths) + spatial_offset,
                      end(b_g_n_c_wis_lengths),
                      begin(input_spatial_lengths_));
            std::copy(begin(e_g_k_c_xs_lengths) + spatial_offset,
                      end(e_g_k_c_xs_lengths),
                      begin(filter_spatial_lengths_));
            std::copy(begin(a_g_n_k_wos_lengths) + spatial_offset,
                      end(a_g_n_k_wos_lengths),
                      begin(output_spatial_lengths_));
        }

        std::size_t GetWorkspaceSizeBytes() const
        {
            return 0;
        }

        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        EDataType* p_e_grid_;

        index_t M01_;
        index_t N01_;

        OutElementwiseOperation a_element_op_;
        InElementwiseOperation b_element_op_;
        WeiElementwiseOperation cde_element_op_;

        // for checking IsSupportedArgument()
        const index_t Conv_G_;
        const index_t Conv_N_;
        const index_t Conv_K_;
        const index_t Conv_C_;
        std::array<ck::index_t, NDimSpatial> input_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> filter_spatial_lengths_;
        std::array<ck::index_t, NDimSpatial> output_spatial_lengths_;
        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides_;
        const std::array<ck::index_t, NDimSpatial>& input_left_pads_;
        const std::array<ck::index_t, NDimSpatial>& input_right_pads_;
        const index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument& arg)
        {

        }
        index_t CalculateGridSize(const Argument& arg)
        {
            return arg.Conv_G_;
        }

        float RunGemmV3(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
    
            index_t gdx = CalculateGridSize(arg);

            float ave_time = 0;

            constexpr index_t minimum_occupancy =
                BlkGemmPipeSched == BlockGemmPipelineScheduler::Intrawave ? 1 : 2;

            const auto kernel = kernel_grouped_conv_bwd_weight_naive<
                GridwiseGemm,
                remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                remove_reference_t<
                    DeviceOp::CGridDesc_MBlock_MPerBlock_NBlock_NPerBlock>,
                ComputePtrOffsetOfStridedBatch<I1, I1, I0>,
                NumGroupsToMerge,
                true,
                InMemoryDataOperationEnum::Set,
                minimum_occupancy>;

            ave_time += launch_and_time_kernel(
                        stream_config, 
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        gemm_arg,
                        arg.a_grid_desc_k0_m_k1_,
                        arg.b_grid_desc_k0_n_k1_,
                        arg.c_grid_desc_mblock_mperblock_nblock_nperblock_,
                        arg.compute_ptr_offset_of_batch_,
                        num_k_per_block);

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

    static bool IsSupportedArgument(const Argument& arg)
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
