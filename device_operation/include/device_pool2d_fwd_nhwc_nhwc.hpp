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
          ReduceTensorOp_t ReduceOp,
          typename InElementwiseOperation,
          typename OutElementwiseOperation,
          ck::index_t BlockSize,
          ck::index_t ReduceMPerBlock,
          ck::index_t ReduceKPerBlock,
          ck::index_t ReduceMPerThread,
          ck::index_t ReduceKPerThread>
struct DevicePool2dFwd_Input_N_Hi_Wi_C_Output_N_Ho_Wo_C
    : public DevicePoolFwd<InDataType,
                           OutDataType,
                           ReduceOp,
                           InElementwiseOperation,
                           OutElementwiseOperation>
{
    using DeviceOp = DevicePool2dFwd_Input_N_Hi_Wi_C_Output_N_Ho_Wo_C;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    static auto MakeABGridDescriptor_A_M_K_B_M(ck::index_t N,
                                               ck::index_t C,
                                               std::vector<ck::index_t> input_spatial_lengths,
                                               std::vector<ck::index_t> window_spatial_lengths,
                                               std::vector<ck::index_t> output_spatial_lengths,
                                               std::vector<ck::index_t> window_strides,
                                               std::vector<ck::index_t> input_left_pads,
                                               std::vector<ck::index_t> input_right_pads)
    {
        using namespace ck;

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
        const index_t ReduceM    = math::integer_least_multiple(ReduceMRaw, ReduceMPerBlock);
        const index_t ReduceMPad = ReduceM - ReduceMRaw;

        const index_t ReduceKRaw = Y * X;
        const index_t ReduceK    = math::integer_least_multiple(ReduceKRaw, ReduceKPerBlock);
        const index_t ReduceKPad = ReduceK - ReduceKRaw;

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
        Argument(const InDataType* p_in_grid,
                 OutDataType* p_out_grid,
                 ck::index_t N,
                 ck::index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> window_spatial_lengths,
                 std::vector<ck::index_t> output_spatial_lengths,
                 std::vector<ck::index_t> window_strides,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 OutElementwiseOperation out_element_op)
            : p_in_grid_{p_in_grid}, p_out_grid_{p_out_grid}, a_grid_desc_m_k_{}, b_grid_desc_m_{}
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
        }

        const InDataType* p_in_grid_;
        OutDataType* p_out_grid_;
        AGridDesc_M_K a_grid_desc_m_k_;
        BGridDesc_M b_grid_desc_m_;
        InElementwiseOperation in_element_op_;
        OutElementwiseOperation out_element_op_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
#if 0
            {
                std::cout << "arg.a_grid_desc_m_k_{" << arg.a_grid_desc_m_k_.GetLength(I0) << ", "
                          << arg.a_grid_desc_m_k_.GetLength(I1) << "} " << std::endl;

                std::cout << "arg.b_grid_desc_m_{" << arg.b_grid_desc_m_.GetLength(I0) << "} "
                          << std::endl;
            }
#endif

            constexpr ck::index_t ThreadPerBlock_ReduceM = ReduceMPerBlock / ReduceMPerThread;
            constexpr ck::index_t ThreadPerBlock_ReduceK = ReduceKPerBlock / ReduceKPerThread;

            using gridwise_reduce =
                GridwiseReduction_xy_to_x_threadwise<InDataType,
                                                     OutDataType,
                                                     AccDataType,
                                                     AGridDesc_M_K,
                                                     BGridDesc_M,
                                                     ReduceOp,
                                                     NanPropagation_t::NOT_PROPAGATE_NAN,
                                                     ReduceTensorIndices_t::NO_INDICES,
                                                     BlockSize,
                                                     ThreadPerBlock_ReduceM,
                                                     ThreadPerBlock_ReduceK,
                                                     ReduceMPerThread,
                                                     ReduceKPerThread,
                                                     true, // dim0_is_fastest
                                                     1,    // dim0_vector_size
                                                     1>;   // dim1_vector_size

            constexpr bool need_indices = false;

            constexpr int RunId = need_indices ? 2 : 1;

            const auto kernel = kernel_reduce_threadwise<gridwise_reduce,
                                                         RunId,
                                                         InDataType,
                                                         OutDataType,
                                                         AGridDesc_M_K,
                                                         BGridDesc_M>;

            ck::index_t ReduceM = arg.a_grid_desc_m_k_.GetLength(I0);
            ck::index_t ReduceK = arg.a_grid_desc_m_k_.GetLength(I1);

            const index_t grid_size = (ReduceM / ReduceMPerBlock);

            return launch_and_time_kernel(kernel,
                                          nrepeat,
                                          dim3(grid_size),
                                          dim3(BlockSize),
                                          0,
                                          arg.a_grid_desc_m_k_,
                                          arg.b_grid_desc_m_,
                                          ReduceK,
                                          float(1),
                                          arg.p_in_grid_,
                                          float(0),
                                          arg.p_out_grid_,
                                          nullptr);
        }

        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        }
    };

    static bool IsSupportedArgument(const Argument& arg) { return true; }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in_grid,
                             OutDataType* p_out_grid,
                             ck::index_t N,
                             ck::index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> window_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> window_strides,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in_grid,
                        p_out_grid,
                        N,
                        C,
                        input_spatial_lengths,
                        window_spatial_lengths,
                        output_spatial_lengths,
                        window_strides,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        out_element_op};
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const InDataType* p_in_grid,
                        OutDataType* p_out_grid,
                        ck::index_t N,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> window_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> window_strides,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        OutElementwiseOperation out_element_op) override
    {
        return std::make_unique<Argument>(MakeArgument(p_in_grid,
                                                       p_out_grid,
                                                       N,
                                                       C,
                                                       input_spatial_lengths,
                                                       window_spatial_lengths,
                                                       output_spatial_lengths,
                                                       window_strides,
                                                       input_left_pads,
                                                       input_right_pads,
                                                       in_element_op,
                                                       out_element_op));
    }

    static auto MakeInvoker() { return Invoker{}; }

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
