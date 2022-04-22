#ifndef DEVICE_BNORM_FWD_NHWC_C_WITH_REDUCE_BLOCKWISE_HPP
#define DEVICE_BNORM_FWD_NHWC_C_WITH_REDUCE_BLOCKWISE_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_bnorm_fwd.hpp"
#include "gridwise_2d_compute_mean_and_meansquare_using_reduce_blockwise.hpp"
#include "gridwise_1d_compute_inv_variance_running_mean_and_variance_fused.hpp"
#include "gridwise_1d_binary_operate.hpp"
#include "gridwise_2d_normalize_blockwise.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename InOutDataType,
          typename AccDataType,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t InOutVectorSize,
          index_t ScaleBiasMeanVarVectorSize>
struct DeviceBatchNormFwd_Input_N_H_W_C_Output_C_With_Reduce_Blockwise : public DeviceBatchNormFwd
{
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert((MThreadSliceSize % InOutVectorSize == 0) &&
                      (MThreadSliceSize % ScaleBiasMeanVarVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr int M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr int K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    long_index_t GetWorkspaceSizeInBytes(index_t c, bool resultSave) override
    {
        // if resultSaveMean and resultSaveInvVariance are available, we use
        // them as workspace directly
        if(resultSave)
        {
            return (0);
        }
        else
        {
            long_index_t wsSizeInBytes = c * sizeof(AccDataType) * 2 + 64;

            return (wsSizeInBytes);
        };
    };

    static auto MakeInOut2dDescriptor(index_t n, index_t h, index_t w, index_t c)
    {
        const auto tupleLengths = make_tuple(n, h, w, c);

        const auto in_out_grid_desc = make_naive_tensor_descriptor_packed(tupleLengths);

        const auto in_out_grid_desc_m_k =
            transform_tensor_descriptor(in_out_grid_desc,
                                        make_tuple(make_merge_transform(make_tuple(c)),
                                                   make_merge_transform(make_tuple(n, h, w))),
                                        make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto length_m = in_out_grid_desc_m_k.GetLength(Number<0>{});
        const auto length_k = in_out_grid_desc_m_k.GetLength(Number<1>{});

        const auto pad_m = math::integer_least_multiple(length_m, M_BlockTileSize) - length_m;
        const auto pad_k = math::integer_least_multiple(length_k, K_BlockTileSize) - length_k;

        auto in_out_grid_desc_m_k_padded =
            transform_tensor_descriptor(in_out_grid_desc_m_k,
                                        make_tuple(make_right_pad_transform(length_m, pad_m),
                                                   make_right_pad_transform(length_k, pad_k)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in_out_grid_desc_m_k_padded);
    };

    static auto MakeScaleBiasMeanVar1dDescriptor(index_t c)
    {
        const auto tupleLengths = make_tuple(c);

        auto scale_bias_mean_var_grid_desc_m = make_naive_tensor_descriptor_packed(tupleLengths);

        const auto length_m = scale_bias_mean_var_grid_desc_m.GetLength(Number<0>{});

        const auto pad_m = math::integer_least_multiple(length_m, M_BlockTileSize) - length_m;

        auto scale_bias_mean_var_grid_desc_m_padded =
            transform_tensor_descriptor(scale_bias_mean_var_grid_desc_m,
                                        make_tuple(make_right_pad_transform(length_m, pad_m)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (scale_bias_mean_var_grid_desc_m_padded);
    };

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<index_t> inLengths,
                 const std::vector<index_t> inStrides,
                 const std::vector<index_t> outLengths,
                 const std::vector<index_t> outStrides,
                 const std::vector<index_t> bnScaleBiasMeanVarLengths,
                 const std::vector<index_t> bnScaleBiasMeanVarStrides,
                 float alpha,
                 float beta,
                 const InOutDataType* in_dev,
                 InOutDataType* out_dev,
                 AccDataType* workspace_dev,
                 const AccDataType* bnScale,
                 const AccDataType* bnBias,
                 double exponentialAverageFactor,
                 AccDataType* resultRunningMean,
                 AccDataType* resultRunningVariance,
                 double epsilon,
                 AccDataType* resultSaveMean,
                 AccDataType* resultSaveInvVariance)
            : in_dev_(in_dev),
              out_dev_(out_dev),
              bnScale_(bnScale),
              bnBias_(bnBias),
              resultRunningMean_(resultRunningMean),
              resultRunningVariance_(resultRunningVariance),
              exponentialAverageFactor_(exponentialAverageFactor),
              epsilon_(epsilon)
        {
            (void)inStrides;
            (void)outStrides;
            (void)bnScaleBiasMeanVarStrides;
            (void)workspace_dev;

            if(inLengths.size() != 4 || outLengths.size() != 4 ||
               bnScaleBiasMeanVarLengths.size() != 1)
                throw std::runtime_error("Invalid tensor dimensions!");

            n = inLengths[0];
            h = inLengths[1];
            w = inLengths[2];
            c = inLengths[3];

            if(outLengths[3] != c || bnScaleBiasMeanVarLengths[0] != c)
                throw std::runtime_error("Inconsistent tensor lengths!");

            alpha_ = type_convert<AccDataType>(alpha);
            beta_  = type_convert<AccDataType>(beta);

            resultSave    = (resultSaveMean != nullptr && resultSaveInvVariance != nullptr);
            resultRunning = (resultRunningMean != nullptr && resultRunningVariance != nullptr);

            if(resultSave)
            {
                resultSaveMean_        = resultSaveMean;
                resultSaveInvVariance_ = resultSaveInvVariance;
            }
            else
            {
                resultSaveMean_        = workspace_dev;
                int alignedSizeInBytes = math::integer_least_multiple(c * sizeof(AccDataType), 64);
                resultSaveInvVariance_ = reinterpret_cast<AccDataType*>(
                    reinterpret_cast<char*>(workspace_dev) + alignedSizeInBytes);
            };

            gridSize = math::integer_least_multiple(c, M_BlockTileSize) / M_BlockTileSize;

            gridSize_2 = math::integer_least_multiple(c, BlockSize) / BlockSize;
        }

        const InOutDataType* in_dev_;
        InOutDataType* out_dev_;
        const AccDataType* bnScale_;
        const AccDataType* bnBias_;
        AccDataType* resultRunningMean_;
        AccDataType* resultRunningVariance_;
        AccDataType* resultSaveMean_;
        AccDataType* resultSaveInvVariance_;

        bool resultSave, resultRunning;

        index_t n, h, w, c;

        AccDataType alpha_, beta_;

        double exponentialAverageFactor_;
        double epsilon_;

        size_t gridSize;
        size_t gridSize_2;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, int nrepeat = 1)
        {
            const auto in_out_grid_desc_m_k =
                DeviceBatchNormFwd_Input_N_H_W_C_Output_C_With_Reduce_Blockwise::
                    MakeInOut2dDescriptor(arg.n, arg.h, arg.w, arg.c);
            const auto scale_bias_mean_var_grid_desc_m =
                DeviceBatchNormFwd_Input_N_H_W_C_Output_C_With_Reduce_Blockwise::
                    MakeScaleBiasMeanVar1dDescriptor(arg.c);
            using InOutGridDesc_M_K          = decltype(in_out_grid_desc_m_k);
            using ScaleBiasMeanVarGridDesc_M = decltype(scale_bias_mean_var_grid_desc_m);

            using ReductionOperation = ck::reduce::Add<AccDataType>;

            using InElementwiseOperationMean =
                ck::tensor_operation::element_wise::UnaryIdentic<AccDataType, AccDataType, false>;
            using AccElementwiseOperationMean =
                ck::tensor_operation::element_wise::UnaryIdentic<AccDataType, AccDataType, true>;

            using InElementwiseOperationMeanSquare =
                ck::tensor_operation::element_wise::UnarySquare<AccDataType, AccDataType, false>;
            using AccElementwiseOperationMeanSquare =
                ck::tensor_operation::element_wise::UnaryIdentic<AccDataType, AccDataType, true>;

            using BinaryOperationInvVariance   = ck::tensor_operation::element_wise::InvVariance;
            using BinaryOperationMovingAverage = ck::tensor_operation::element_wise::MovingAverage;
            using TernaryOperationNormalize    = ck::tensor_operation::element_wise::Normalize;

            using GridwiseReduceMean =
                GridwiseReduction_mk_to_m_blockwise<InOutDataType,
                                                    AccDataType,
                                                    AccDataType,
                                                    int32_t, // not used
                                                    InOutGridDesc_M_K,
                                                    ScaleBiasMeanVarGridDesc_M,
                                                    ReductionOperation,
                                                    InElementwiseOperationMean,
                                                    AccElementwiseOperationMean,
                                                    false, // PropagateNan
                                                    true,  // BetaIsZero
                                                    BlockSize,
                                                    MThreadClusterSize,
                                                    KThreadClusterSize,
                                                    MThreadSliceSize,
                                                    KThreadSliceSize,
                                                    0, // InSrcVectorDim
                                                    InOutVectorSize,
                                                    ScaleBiasMeanVarVectorSize>;

            using GridwiseReduceMeanSquare =
                GridwiseReduction_mk_to_m_blockwise<InOutDataType,
                                                    AccDataType,
                                                    AccDataType,
                                                    int32_t, // not used
                                                    InOutGridDesc_M_K,
                                                    ScaleBiasMeanVarGridDesc_M,
                                                    ReductionOperation,
                                                    InElementwiseOperationMeanSquare,
                                                    AccElementwiseOperationMeanSquare,
                                                    false, // PropagateNan
                                                    true,  // BetaIsZero
                                                    BlockSize,
                                                    MThreadClusterSize,
                                                    KThreadClusterSize,
                                                    MThreadSliceSize,
                                                    KThreadSliceSize,
                                                    0, // InSrcVectorDim
                                                    InOutVectorSize,
                                                    ScaleBiasMeanVarVectorSize>;

            InElementwiseOperationMean in_element_wise_op_mean{};
            AccElementwiseOperationMean acc_element_wise_op_mean{arg.n * arg.h * arg.w};
            InElementwiseOperationMeanSquare in_element_wise_op_meansquare{};
            AccElementwiseOperationMeanSquare acc_element_wise_op_meansquare{arg.n * arg.h * arg.w};

            const auto kernel_mean_and_meansquare =
                kernel_compute_mean_and_meansquare_using_reduction_blockwise<
                    GridwiseReduceMean,
                    GridwiseReduceMeanSquare,
                    InOutDataType,
                    AccDataType,
                    InOutGridDesc_M_K,
                    ScaleBiasMeanVarGridDesc_M,
                    InElementwiseOperationMean,
                    AccElementwiseOperationMean,
                    InElementwiseOperationMeanSquare,
                    AccElementwiseOperationMeanSquare>;

            using GridwiseNormalize = GridwiseNormalizeBlockwise_mk_input_m_scale_bias_mean_var<
                InOutDataType,
                InOutDataType,
                AccDataType,
                InOutGridDesc_M_K,
                ScaleBiasMeanVarGridDesc_M,
                TernaryOperationNormalize,
                BlockSize,
                MThreadClusterSize,
                KThreadClusterSize,
                MThreadSliceSize,
                KThreadSliceSize,
                0, // InOutVectorDim,
                InOutVectorSize,
                ScaleBiasMeanVarVectorSize>;

            const auto kernel_normalize = kernel_normalize_blockwise<GridwiseNormalize,
                                                                     InOutDataType,
                                                                     InOutDataType,
                                                                     AccDataType,
                                                                     InOutGridDesc_M_K,
                                                                     ScaleBiasMeanVarGridDesc_M,
                                                                     TernaryOperationNormalize>;

            float avg_time = 0.0f;

            KernelTimer timer;

            for(int i = 0; i < nrepeat + 1; i++)
            {
                if(i == 1)
                    timer.Start();

                launch_kernel(kernel_mean_and_meansquare,
                              dim3(arg.gridSize),
                              dim3(BlockSize),
                              0,
                              in_out_grid_desc_m_k,
                              scale_bias_mean_var_grid_desc_m,
                              in_element_wise_op_mean,
                              acc_element_wise_op_mean,
                              in_element_wise_op_meansquare,
                              acc_element_wise_op_meansquare,
                              arg.in_dev_,
                              arg.resultSaveMean_,         // mean values
                              arg.resultSaveInvVariance_); // meansquare values

                if(arg.resultRunning)
                {
                    BinaryOperationInvVariance op_invVariance(arg.epsilon_);
                    BinaryOperationMovingAverage op_movingAverage(arg.exponentialAverageFactor_);

                    const auto kernel_inv_variance_running_mean_and_variance =
                        kernel_1d_compute_inv_variance_running_mean_and_variance<
                            BlockSize,
                            AccDataType,
                            ScaleBiasMeanVarGridDesc_M,
                            BinaryOperationInvVariance,
                            BinaryOperationMovingAverage>;

                    launch_kernel(kernel_inv_variance_running_mean_and_variance,
                                  dim3(arg.gridSize_2),
                                  dim3(BlockSize),
                                  0,
                                  scale_bias_mean_var_grid_desc_m,
                                  op_invVariance,
                                  op_movingAverage,
                                  arg.resultSaveMean_,        // mean values
                                  arg.resultSaveInvVariance_, // meansquare values
                                  arg.resultSaveInvVariance_,
                                  arg.resultRunningMean_,
                                  arg.resultRunningVariance_);
                }
                else
                {
                    BinaryOperationInvVariance op_invVariance(arg.epsilon_);

                    const auto kernel_inv_variance =
                        kernel_1d_binary_operate<BlockSize,
                                                 AccDataType,
                                                 AccDataType,
                                                 AccDataType,
                                                 AccDataType,
                                                 ScaleBiasMeanVarGridDesc_M,
                                                 BinaryOperationInvVariance>;

                    launch_kernel(kernel_inv_variance,
                                  dim3(arg.gridSize_2),
                                  dim3(BlockSize),
                                  0,
                                  scale_bias_mean_var_grid_desc_m,
                                  op_invVariance,
                                  arg.resultSaveMean_,        // mean values
                                  arg.resultSaveInvVariance_, // meansquare values
                                  arg.resultSaveInvVariance_);
                };

                TernaryOperationNormalize op_normalize;

                launch_kernel(kernel_normalize,
                              dim3(arg.gridSize),
                              dim3(BlockSize),
                              0,
                              in_out_grid_desc_m_k,
                              scale_bias_mean_var_grid_desc_m,
                              op_normalize,
                              arg.in_dev_,
                              arg.out_dev_,
                              arg.alpha_,
                              arg.beta_,
                              arg.bnScale_,
                              arg.bnBias_,
                              arg.resultSaveMean_,
                              arg.resultSaveInvVariance_);
            };

            timer.End();

            avg_time = timer.GetElapsedTime() / nrepeat;

            return (avg_time);
        };

        float Run(const BaseArgument* p_arg, int nrepeat = 1) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), nrepeat);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if(pArg->c % InOutVectorSize != 0)
            return (false);

        // To improve
        if(pArg->c % ScaleBiasMeanVarVectorSize != 0)
            return (false);

        return (true);
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> inLengths,
                        const std::vector<index_t> inStrides,
                        const std::vector<index_t> outLengths,
                        const std::vector<index_t> outStrides,
                        const std::vector<index_t> bnScaleBiasMeanVarLengths,
                        const std::vector<index_t> bnScaleBiasMeanVarStrides,
                        float alpha,
                        float beta,
                        const void* in_dev,
                        void* out_dev,
                        void* workspace_dev,
                        const void* bnScale,
                        const void* bnBias,
                        double exponentialAverageFactor,
                        void* resultRunningMean,
                        void* resultRunningVariance,
                        double epsilon,
                        void* resultSaveMean,
                        void* resultSaveInvVariance) override
    {
        return std::make_unique<Argument>(inLengths,
                                          inStrides,
                                          outLengths,
                                          outStrides,
                                          bnScaleBiasMeanVarLengths,
                                          bnScaleBiasMeanVarStrides,
                                          alpha,
                                          beta,
                                          static_cast<const InOutDataType*>(in_dev),
                                          static_cast<InOutDataType*>(out_dev),
                                          static_cast<AccDataType*>(workspace_dev),
                                          static_cast<const AccDataType*>(bnScale),
                                          static_cast<const AccDataType*>(bnBias),
                                          exponentialAverageFactor,
                                          static_cast<AccDataType*>(resultRunningMean),
                                          static_cast<AccDataType*>(resultRunningVariance),
                                          epsilon,
                                          static_cast<AccDataType*>(resultSaveMean),
                                          static_cast<AccDataType*>(resultSaveInvVariance));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchNorm_NHWC_C_Blockwise<" << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ",";
        str << "InOutVectorSize_" << InOutVectorSize << "_ScaleBiasMeanVarVectorSize_" << ScaleBiasMeanVarVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
