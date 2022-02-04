#ifndef DEVICE_POOL2D_FWD_NHWC_NHWC_HPP
#define DEVICE_POOL2D_FWD_NHWC_NHWC_HPP

#include <iostream>
#include <sstream>
#include "device_pool_fwd.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_2d_reduction_threadwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InDataType,
          typename OutDataType,
          typename AccDataType,
          ck::ReduceTensorOp_t ReduceOpId,
          bool NeedIndices,
          ck::index_t BlockSize,
          ck::index_t ReduceMThreadClusterSize,
          ck::index_t ReduceKThreadClusterSize,
          ck::index_t ReduceMThreadSliceSize,
          ck::index_t ReduceKThreadSliceSize,
          ck::index_t InSrcOutDstVectorSize>
struct DevicePool2dFwd_Input_N_Hi_Wi_C_Output_N_Ho_Wo_C : public DevicePoolFwd<ReduceOpId>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    using ReduceOperation = typename reduce_binary_operator<AccDataType, ReduceOpId>::opType;

    using InElementwiseOperation =
        typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::InElementwiseOperation;

    using AccElementwiseOperation =
        typename reduce_unary_operator<AccDataType, ReduceOpId, true, true>::
            AccElementwiseOperation;

    static constexpr bool BetaIsZero = true;

    static constexpr int InSrcOutDstVectorDim =
        0; // for NHWC, the dim C is the vector Dim for both input and output in memory, which is
           // not reduced.

    static constexpr ck::index_t ReduceM_BlockTileSize =
        ReduceMThreadClusterSize * ReduceMThreadSliceSize;
    static constexpr ck::index_t ReduceK_BlockTileSize =
        ReduceKThreadClusterSize * ReduceKThreadSliceSize;

    static auto MakeABGridDescriptor_A_M_K_B_M(ck::index_t N,
                                               ck::index_t C,
                                               std::array<ck::index_t, 2> input_spatial_lengths,
                                               std::array<ck::index_t, 2> window_spatial_lengths,
                                               std::array<ck::index_t, 2> output_spatial_lengths,
                                               std::array<ck::index_t, 2> window_strides,
                                               std::array<ck::index_t, 2> input_left_pads,
                                               std::array<ck::index_t, 2> input_right_pads)
    {
        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t Y = window_spatial_lengths[0];
        const index_t X = window_spatial_lengths[1];

        const index_t ConvStrideH = window_strides[0];
        const index_t ConvStrideW = window_strides[1];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t ReduceMRaw = N * Ho * Wo * C;
        const index_t ReduceMPad =
            math::integer_least_multiple(ReduceMRaw, ReduceM_BlockTileSize) - ReduceMRaw;

        const index_t ReduceKRaw = Y * X;
        const index_t ReduceKPad =
            math::integer_least_multiple(ReduceKRaw, ReduceK_BlockTileSize) - ReduceKRaw;

        // A[ReduceM, ReduceK]
        const auto in_grid_desc_n_hi_wi_c =
            make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

        const auto in_grid_desc_n_hip_wip_c = transform_tensor_descriptor(
            in_grid_desc_n_hi_wi_c,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto in_grid_desc_n_y_ho_x_wo_c = transform_tensor_descriptor(
            in_grid_desc_n_hip_wip_c,
            make_tuple(make_pass_through_transform(N),
                       make_embed_transform(make_tuple(Y, Ho), make_tuple(I1, ConvStrideH)),
                       make_embed_transform(make_tuple(X, Wo), make_tuple(I1, ConvStrideW)),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        const auto in_grid_desc_reducemraw_reducekraw =
            transform_tensor_descriptor(in_grid_desc_n_y_ho_x_wo_c,
                                        make_tuple(make_merge_transform(make_tuple(N, Ho, Wo, C)),
                                                   make_merge_transform(make_tuple(Y, X))),
                                        make_tuple(Sequence<0, 2, 4, 5>{}, Sequence<1, 3>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto in_grid_desc_reducem_reducek = transform_tensor_descriptor(
            in_grid_desc_reducemraw_reducekraw,
            make_tuple(make_right_pad_transform(ReduceMRaw, ReduceMPad),
                       make_right_pad_transform(ReduceKRaw, ReduceKPad)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        // B[ReduceM]
        const auto out_grid_desc_reducemraw =
            make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo * C));

        const auto out_grid_desc_reducem = transform_tensor_descriptor(
            out_grid_desc_reducemraw,
            make_tuple(make_right_pad_transform(ReduceMRaw, ReduceMPad)),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0>{}));

        return make_tuple(in_grid_desc_reducem_reducek, out_grid_desc_reducem);
    }

    using ABGridDescs = decltype(
        MakeABGridDescriptor_A_M_K_B_M(1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}));

    using AGridDesc_M_K = remove_cvref_t<decltype(ABGridDescs{}[I0])>;
    using BGridDesc_M   = remove_cvref_t<decltype(ABGridDescs{}[I1])>;

    // TODO
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_dev,
                 OutDataType* p_out_dev,
                 int* p_out_indices_dev,
                 ck::index_t N,
                 ck::index_t C,
                 std::array<ck::index_t, 2>& input_spatial_lengths,
                 std::array<ck::index_t, 2>& window_spatial_lengths,
                 std::array<ck::index_t, 2>& output_spatial_lengths,
                 std::array<ck::index_t, 2>& window_strides,
                 std::array<ck::index_t, 2>& input_left_pads,
                 std::array<ck::index_t, 2>& input_right_pads)
            : p_in_dev_{p_in_dev},
              p_out_dev_{p_out_dev},
              p_out_indices_dev_{p_out_indices_dev},
              a_grid_desc_m_k_{},
              b_grid_desc_m_{}
        {
            const auto descs = MakeABGridDescriptor_A_M_K_B_M(N,
                                                              C,
                                                              input_spatial_lengths,
                                                              window_spatial_lengths,
                                                              output_spatial_lengths,
                                                              window_strides,
                                                              input_left_pads,
                                                              input_right_pads);

            a_grid_desc_m_k_ = descs[I0];
            b_grid_desc_m_   = descs[I1];

            outer_lowest_length_ = C;
            inner_lowest_length_ = window_spatial_lengths[1];

            // TODO: is this correct?
            if constexpr(ReduceOpId == ck::ReduceTensorOp_t::AVG)
            {
                ck::index_t divider = window_spatial_lengths[0] * window_spatial_lengths[1];
                in_element_op_      = InElementwiseOperation{divider};
                acc_element_op_     = AccElementwiseOperation{divider};
            }
        }

        const InDataType* p_in_dev_;
        OutDataType* p_out_dev_;
        int* p_out_indices_dev_;
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_M b_grid_desc_m_;
        InElementwiseOperation in_element_op_;
        AccElementwiseOperation acc_element_op_;

        // for checking vector load/store
        ck::index_t outer_lowest_length_;
        ck::index_t inner_lowest_length_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            using gridwise_reduce = GridwiseReduction_xy_to_x_threadwise<InDataType,
                                                                         OutDataType,
                                                                         AccDataType,
                                                                         AGridDesc_M_K,
                                                                         BGridDesc_M,
                                                                         ReduceOperation,
                                                                         InElementwiseOperation,
                                                                         AccElementwiseOperation,
                                                                         false, // propagate_nan
                                                                         BetaIsZero,
                                                                         BlockSize,
                                                                         ReduceMThreadClusterSize,
                                                                         ReduceKThreadClusterSize,
                                                                         ReduceMThreadSliceSize,
                                                                         ReduceKThreadSliceSize,
                                                                         InSrcOutDstVectorDim,
                                                                         InSrcOutDstVectorSize,
                                                                         InSrcOutDstVectorSize>;

            const auto kernel = kernel_reduce_threadwise<gridwise_reduce,
                                                         NeedIndices,
                                                         InDataType,
                                                         OutDataType,
                                                         AccDataType,
                                                         AGridDesc_M_K,
                                                         BGridDesc_M,
                                                         InElementwiseOperation,
                                                         AccElementwiseOperation>;

            ck::index_t ReduceM = arg.a_grid_desc_m_k_.GetLength(I0);

            const index_t grid_size = (ReduceM / ReduceM_BlockTileSize);

            return launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          arg.a_grid_desc_m_k_,
                                          arg.b_grid_desc_m_,
                                          arg.in_element_op_,
                                          arg.acc_element_op_,
                                          float(1),
                                          arg.p_in_dev_,
                                          float(0),
                                          arg.p_out_dev_,
                                          arg.p_out_indices_dev_);
        }

        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        }
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if(pArg->outer_lowest_length_ % InSrcOutDstVectorSize != 0)
        {
            return (false);
        }

        return (true);
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_dev,
                        void* p_out_dev,
                        void* p_out_indices_dev,
                        ck::index_t N,
                        ck::index_t C,
                        std::array<ck::index_t, 2> input_spatial_lengths,
                        std::array<ck::index_t, 2> window_spatial_lengths,
                        std::array<ck::index_t, 2> output_spatial_lengths,
                        std::array<ck::index_t, 2> window_strides,
                        std::array<ck::index_t, 2> input_left_pads,
                        std::array<ck::index_t, 2> input_right_pads) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_dev),
                                          static_cast<OutDataType*>(p_out_dev),
                                          static_cast<int*>(p_out_indices_dev),
                                          N,
                                          C,
                                          input_spatial_lengths,
                                          window_spatial_lengths,
                                          output_spatial_lengths,
                                          window_strides,
                                          input_left_pads,
                                          input_right_pads);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DevicePool2dFwd_Input_N_Hi_Wi_C_Output_N_Ho_Wo_C"
            << "<"
            << ">";
        // clang-format on

        return str.str();
    }
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
