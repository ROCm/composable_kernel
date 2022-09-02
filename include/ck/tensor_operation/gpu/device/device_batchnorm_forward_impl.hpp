// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/device_batchnorm_forward.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_multiblock_welford_first_half.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_multiblock_welford_second_half_batchnorm_forward_final.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batchnorm_forward_blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_set_buffer_value.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/utility/host_common_util.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename XDataType,
          typename YDataType,
          typename AccDataType,
          typename ScaleBiasDataType,
          typename MeanVarDataType,
          index_t Rank,
          index_t NumBatchNormReduceDim,
          bool UseMultiblockInK,
          bool PropagateNan,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t XSrcYDstVectorDim,
          index_t XSrcVectorSize,
          index_t YDstVectorSize,
          index_t ScaleBiasSrcVectorSize,
          index_t MeanVarSrcDstVectorSize>
struct DeviceBatchNormFwdImpl : public DeviceBatchNormFwd<Rank, NumBatchNormReduceDim>
{
    static_assert(Rank <= 6, "Bigger Rank size is not supported!");
    static_assert(BlockSize == MThreadClusterSize * KThreadClusterSize,
                  "Invalid thread cluster size assignments!");

    static_assert((XSrcYDstVectorDim == 0 && MThreadSliceSize % XSrcVectorSize == 0) ||
                      (XSrcYDstVectorDim == 1 && KThreadSliceSize % XSrcVectorSize == 0),
                  "Invalid thread slice sizes and/or vector sizes configuration, please check!");

    static constexpr index_t NumInvariantDim = Rank - NumBatchNormReduceDim;

    static constexpr index_t M_BlockTileSize = MThreadClusterSize * MThreadSliceSize;
    static constexpr index_t K_BlockTileSize = KThreadClusterSize * KThreadSliceSize;

    static auto MakeXY2dDescriptor(const std::array<index_t, Rank>& xyLengths,
                                   const std::array<index_t, Rank>& xyStrides,
                                   int blkGroupSize,
                                   int numBlockTileIteration)
    {
        const auto tupleXYLengths =
            generate_tuple([&](auto I) { return xyLengths[I]; }, Number<Rank>{});
        const auto tupleXYStrides =
            generate_tuple([&](auto I) { return xyStrides[I]; }, Number<Rank>{});

        const auto raw_grid_desc = make_naive_tensor_descriptor(tupleXYLengths, tupleXYStrides);

        const auto grid_desc_m_k = [&]() {
            using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;
            using ReduceDims    = typename arithmetic_sequence_gen<NumInvariantDim, Rank, 1>::type;

            const auto reduceDimLengths =
                generate_tuple([&](auto I) { return xyLengths[NumInvariantDim + I]; },
                               Number<NumBatchNormReduceDim>{});
            const auto invariantDimLengths =
                generate_tuple([&](auto I) { return xyLengths[I]; }, Number<NumInvariantDim>{});

            return transform_tensor_descriptor(raw_grid_desc,
                                               make_tuple(make_merge_transform(invariantDimLengths),
                                                          make_merge_transform(reduceDimLengths)),
                                               make_tuple(InvariantDims{}, ReduceDims{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }();

        const auto invariantLength = grid_desc_m_k.GetLength(Number<0>{});
        const auto reduceLength    = grid_desc_m_k.GetLength(Number<1>{});

        const int workSizePerBlock = K_BlockTileSize * numBlockTileIteration;
        const auto mPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto kPad = workSizePerBlock * blkGroupSize - reduceLength;

        auto grid_desc_m_k_padded =
            transform_tensor_descriptor(grid_desc_m_k,
                                        make_tuple(make_right_pad_transform(invariantLength, mPad),
                                                   make_right_pad_transform(reduceLength, kPad)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (grid_desc_m_k_padded);
    };

    static auto MakeMeanVarCountOutputMG2dDescriptor(int invariantLength, int blkGroupSize)
    {
        const auto grid_desc_m_g =
            make_naive_tensor_descriptor_packed(make_tuple(invariantLength, blkGroupSize));

        const auto mPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto grid_desc_m_g_padded =
            transform_tensor_descriptor(grid_desc_m_g,
                                        make_tuple(make_right_pad_transform(invariantLength, mPad),
                                                   make_pass_through_transform(blkGroupSize)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (grid_desc_m_g_padded);
    };

    static auto MakeMeanVarCountInputMK2dDescriptor(int invariantLength, int blkGroupSize)
    {
        const auto reduceLength = blkGroupSize;
        const auto grid_desc_m_k =
            make_naive_tensor_descriptor_packed(make_tuple(invariantLength, reduceLength));

        const auto mPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto kPad =
            math::integer_least_multiple(reduceLength, KThreadClusterSize) - reduceLength;

        auto grid_desc_m_k_padded =
            transform_tensor_descriptor(grid_desc_m_k,
                                        make_tuple(make_right_pad_transform(invariantLength, mPad),
                                                   make_right_pad_transform(reduceLength, kPad)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (grid_desc_m_k_padded);
    };

    static auto
    MakeScaleBiasMeanVar1dDescriptor(const std::array<index_t, NumInvariantDim>& lengths,
                                     const std::array<index_t, NumInvariantDim>& strides)
    {
        const auto tupleLengths =
            generate_tuple([&](auto I) { return lengths[I]; }, Number<NumInvariantDim>{});
        const auto tupleStrides =
            generate_tuple([&](auto I) { return strides[I]; }, Number<NumInvariantDim>{});

        auto raw_grid_desc = make_naive_tensor_descriptor(tupleLengths, tupleStrides);

        auto grid_desc_m = transform_tensor_descriptor(
            raw_grid_desc,
            make_tuple(make_merge_transform(tupleLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto invariantLength = grid_desc_m.GetLength(Number<0>{});

        const auto mPad =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;

        auto grid_desc_m_padded =
            transform_tensor_descriptor(grid_desc_m,
                                        make_tuple(make_right_pad_transform(invariantLength, mPad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (grid_desc_m_padded);
    };

    static auto MakeX1dDescriptorForBufferSet(const std::array<index_t, Rank>& xLengths,
                                              const std::array<index_t, Rank>& xStrides)
    {
        const auto tupleXLengths =
            generate_tuple([&](auto I) { return xLengths[I]; }, Number<Rank>{});
        const auto tupleXStrides =
            generate_tuple([&](auto I) { return xStrides[I]; }, Number<Rank>{});

        const auto raw_grid_desc = make_naive_tensor_descriptor(tupleXLengths, tupleXStrides);

        auto grid_desc_m = transform_tensor_descriptor(
            raw_grid_desc,
            make_tuple(make_merge_transform(tupleXLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, Rank, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto length = grid_desc_m.GetLength(Number<0>{});

        const auto pad = math::integer_least_multiple(length, BlockSize) - length;

        auto grid_desc_m_padded =
            transform_tensor_descriptor(grid_desc_m,
                                        make_tuple(make_right_pad_transform(length, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return (grid_desc_m_padded);
    };

    using XYGridDesc_M_K             = decltype(MakeXY2dDescriptor({1}, {1}, 1, 1));
    using ScaleBiasMeanVarGridDesc_M = decltype(MakeScaleBiasMeanVar1dDescriptor({1}, {1}));

    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, Rank> xyLengths,
                 const std::array<index_t, Rank> xStrides,
                 const std::array<index_t, Rank> yStrides,
                 const std::array<int, NumBatchNormReduceDim> reduceDims,
                 const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths,
                 const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasStrides,
                 const std::array<index_t, Rank - NumBatchNormReduceDim> bnMeanVarStrides,
                 const XDataType* p_x,
                 const ScaleBiasDataType* p_scale,
                 const ScaleBiasDataType* p_bias,
                 double epsilon,
                 YDataType* p_y,
                 MeanVarDataType* resultSaveMean,
                 MeanVarDataType* resultSaveInvVariance,
                 double averageFactor,
                 MeanVarDataType* resultRunningMean,
                 MeanVarDataType* resultRunningVariance)
            : bnScaleBiasMeanVarLengths_(bnScaleBiasMeanVarLengths),
              bnScaleBiasStrides_(bnScaleBiasStrides),
              bnMeanVarStrides_(bnMeanVarStrides),
              p_x_(p_x),
              p_scale_(p_scale),
              p_bias_(p_bias),
              p_y_(p_y),
              resultSaveMean_(resultSaveMean),
              resultSaveInvVariance_(resultSaveInvVariance),
              resultRunningMean_(resultRunningMean),
              resultRunningVariance_(resultRunningVariance)
        {
            xyLengths_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(xyLengths, reduceDims);
            xStrides_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(xStrides, reduceDims);
            yStrides_ =
                shuffle_tensor_dimensions<Rank, NumBatchNormReduceDim>(yStrides, reduceDims);

            std::tie(invariant_length, reduce_length) =
                get_2d_lengths<Rank, NumBatchNormReduceDim>(xyLengths_);

            epsilon_       = type_convert<AccDataType>(epsilon);
            averageFactor_ = type_convert<AccDataType>(averageFactor);

            updateMovingAverage =
                (resultRunningMean != nullptr && resultRunningVariance != nullptr);
            saveMeanInvVariance = (resultSaveMean != nullptr && resultSaveInvVariance_ != nullptr);

            if(UseMultiblockInK)
            {
                int iterations = 1;
                while(true)
                {
                    int testBlkGroupSize = (reduce_length + (K_BlockTileSize * iterations) - 1) /
                                           (K_BlockTileSize * iterations);

                    // we want the blkGroupSize be not more than 128
                    if(testBlkGroupSize <= 128)
                        break;

                    iterations++;
                };

                blkGroupSize = (reduce_length + (K_BlockTileSize * iterations) - 1) /
                               (K_BlockTileSize * iterations);

                numBlockTileIteration = iterations;
            }
            else
            {
                blkGroupSize          = 1;
                numBlockTileIteration = (reduce_length + K_BlockTileSize - 1) / K_BlockTileSize;
            };

            gridSize = (invariant_length + M_BlockTileSize - 1) / M_BlockTileSize * blkGroupSize;

            x_grid_desc_m_k =
                MakeXY2dDescriptor(xyLengths_, xStrides_, blkGroupSize, numBlockTileIteration);
            y_grid_desc_m_k =
                MakeXY2dDescriptor(xyLengths_, yStrides_, blkGroupSize, numBlockTileIteration);
            scale_bias_grid_desc_m =
                MakeScaleBiasMeanVar1dDescriptor(bnScaleBiasMeanVarLengths, bnScaleBiasStrides);
            mean_var_grid_desc_m =
                MakeScaleBiasMeanVar1dDescriptor(bnScaleBiasMeanVarLengths, bnMeanVarStrides);
        }

        AccDataType epsilon_;
        AccDataType averageFactor_;

        bool updateMovingAverage;
        bool saveMeanInvVariance;

        std::array<index_t, Rank> xyLengths_;
        std::array<index_t, Rank> xStrides_;
        std::array<index_t, Rank> yStrides_;

        std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths_;
        std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasStrides_;
        std::array<index_t, Rank - NumBatchNormReduceDim> bnMeanVarStrides_;

        const XDataType* p_x_;
        const ScaleBiasDataType* p_scale_;
        const ScaleBiasDataType* p_bias_;
        YDataType* p_y_;

        MeanVarDataType* resultSaveMean_;
        MeanVarDataType* resultSaveInvVariance_;

        MeanVarDataType* resultRunningMean_;
        MeanVarDataType* resultRunningVariance_;

        long_index_t invariant_length;
        long_index_t reduce_length;

        int blkGroupSize;
        int numBlockTileIteration;
        size_t gridSize;

        XYGridDesc_M_K x_grid_desc_m_k;
        XYGridDesc_M_K y_grid_desc_m_k;
        ScaleBiasMeanVarGridDesc_M scale_bias_grid_desc_m;
        ScaleBiasMeanVarGridDesc_M mean_var_grid_desc_m;

        void* workspace_initial_count;
    };

    size_t GetWorkSpaceSize(const BaseArgument* pArg) const override
    {
        const Argument* pArg_ = dynamic_cast<const Argument*>(pArg);

        size_t workspace_size = 0;

        // workspace for welford initial count
        workspace_size += pArg_->x_grid_desc_m_k.GetElementSize() * sizeof(int8_t) + 64;

        if(UseMultiblockInK && pArg_->blkGroupSize > 1)
        {
            // workspace for welford intermediate mean
            workspace_size +=
                pArg_->invariant_length * pArg_->blkGroupSize * sizeof(MeanVarDataType) + 64;

            // workspace for welford intermediate variance
            workspace_size +=
                pArg_->invariant_length * pArg_->blkGroupSize * sizeof(MeanVarDataType) + 64;

            // workspace for welford intermediate count
            workspace_size += pArg_->invariant_length * pArg_->blkGroupSize * sizeof(int32_t) + 64;
        }

        return (workspace_size);
    };

    void SetWorkSpacePointer(BaseArgument* pArg, void* p_workspace) const override
    {
        Argument* pArg_ = dynamic_cast<Argument*>(pArg);

        pArg_->p_workspace_ = p_workspace;

        pArg_->workspace_initial_count = pArg_->p_workspace_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto x_grid_desc_m = DeviceBatchNormFwdImpl::MakeX1dDescriptorForBufferSet(
                arg.xyLengths_, arg.xStrides_);

            const auto kernel_set_initial_count =
                kernel_buffer_set_value<BlockSize, int8_t, decltype(x_grid_desc_m)>;

            // for the set value kernel
            size_t gridSize_0 = (x_grid_desc_m.GetLength(Number<0>{}) + BlockSize - 1) / BlockSize;

            float avg_time = 0;

            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_set_initial_count,
                                               dim3(gridSize_0),
                                               dim3(BlockSize),
                                               0,
                                               x_grid_desc_m,
                                               static_cast<int8_t*>(arg.workspace_initial_count),
                                               1);

            if(UseMultiblockInK && arg.blkGroupSize > 1)
            {
                const auto mean_var_count_grid_desc_m_g =
                    DeviceBatchNormFwdImpl::MakeMeanVarCountOutputMG2dDescriptor(
                        arg.invariant_length, arg.blkGroupSize);

                const auto mean_var_count_grid_desc_m_k =
                    DeviceBatchNormFwdImpl::MakeMeanVarCountInputMK2dDescriptor(
                        arg.invariant_length, arg.blkGroupSize);

                using MeanVarCountGridDesc_M_G = decltype(mean_var_count_grid_desc_m_g);
                using MeanVarCountGridDesc_M_K = decltype(mean_var_count_grid_desc_m_k);

                using GridwiseMultiblockWelfordFirstHalf_ =
                    GridwiseMultiblockWelfordFirstHalf<XDataType,
                                                       AccDataType,
                                                       MeanVarDataType,
                                                       XYGridDesc_M_K,
                                                       MeanVarCountGridDesc_M_G,
                                                       BlockSize,
                                                       MThreadClusterSize,
                                                       KThreadClusterSize,
                                                       MThreadSliceSize,
                                                       KThreadSliceSize,
                                                       XSrcYDstVectorDim,
                                                       XSrcVectorSize>;

                using GridwiseWelfordSecondHalfBatchNormForwardFinal_ =
                    GridwiseWelfordSecondHalfBatchNormForwardFinal<XDataType,
                                                                   YDataType,
                                                                   AccDataType,
                                                                   ScaleBiasDataType,
                                                                   MeanVarDataType,
                                                                   XYGridDesc_M_K,
                                                                   MeanVarCountGridDesc_M_K,
                                                                   ScaleBiasMeanVarGridDesc_M,
                                                                   ScaleBiasMeanVarGridDesc_M,
                                                                   BlockSize,
                                                                   MThreadClusterSize,
                                                                   KThreadClusterSize,
                                                                   MThreadSliceSize,
                                                                   KThreadSliceSize,
                                                                   XSrcYDstVectorDim,
                                                                   XSrcVectorSize,
                                                                   YDstVectorSize,
                                                                   ScaleBiasSrcVectorSize,
                                                                   MeanVarSrcDstVectorSize>;

                index_t numMeanVarCountBlockTileIteration =
                    (arg.blkGroupSize + KThreadClusterSize - 1) / KThreadClusterSize;

                const auto kern_multiblock_welford_first_half =
                    kernel_multiblock_welford_first_half<GridwiseMultiblockWelfordFirstHalf_,
                                                         XDataType,
                                                         AccDataType,
                                                         MeanVarDataType,
                                                         XYGridDesc_M_K,
                                                         MeanVarCountGridDesc_M_G>;

                const auto kern_welford_second_half_batchnorm_forward_final =
                    kernel_welford_second_half_batchnorm_forward_final<
                        GridwiseWelfordSecondHalfBatchNormForwardFinal_,
                        XDataType,
                        YDataType,
                        AccDataType,
                        ScaleBiasDataType,
                        MeanVarDataType,
                        XYGridDesc_M_K,
                        MeanVarCountGridDesc_M_K,
                        ScaleBiasMeanVarGridDesc_M,
                        ScaleBiasMeanVarGridDesc_M>;

                avg_time += launch_and_time_kernel(stream_config,
                                                   kern_multiblock_welford_first_half,
                                                   dim3(arg.gridSize),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.x_grid_desc_m_k,
                                                   mean_var_count_grid_desc_m_g,
                                                   arg.numBlockTileIteration,
                                                   arg.p_x_,
                                                   arg.p_workspace_);

                /*
                                using ck::host_common::dumpBufferToFile;

                                MeanVarDataType* imm_mean = new
                   MeanVarDataType[mean_var_count_grid_desc_m_g.GetElementSize()]; MeanVarDataType*
                   imm_var = new MeanVarDataType[mean_var_count_grid_desc_m_g.GetElementSize()];
                                int32_t* imm_count = new
                   int32_t[mean_var_count_grid_desc_m_g.GetElementSize()];

                                index_t count_space_sz = arg.x_grid_desc_m_k.GetElementSize() *
                   sizeof(int8_t); count_space_sz = math::integer_least_multiple(count_space_sz,
                   64); index_t mean_space_sz = mean_var_count_grid_desc_m_g.GetElementSize() *
                   sizeof(MeanVarDataType); mean_space_sz =
                   math::integer_least_multiple(mean_space_sz, 64); index_t variance_space_sz =
                   mean_var_count_grid_desc_m_g.GetElementSize() *  sizeof(MeanVarDataType);
                                variance_space_sz = math::integer_least_multiple(variance_space_sz,
                   64);

                                (void)hipMemcpy(imm_mean, (char*)arg.workspace_initial_count +
                   count_space_sz, mean_var_count_grid_desc_m_g.GetElementSize() *
                   sizeof(MeanVarDataType), hipMemcpyDeviceToHost); (void)hipMemcpy(imm_var,
                   (char*)arg.workspace_initial_count + count_space_sz + mean_space_sz,
                                                mean_var_count_grid_desc_m_g.GetElementSize() *
                   sizeof(MeanVarDataType), hipMemcpyDeviceToHost); (void)hipMemcpy(imm_count,
                   (char*)arg.workspace_initial_count + count_space_sz + mean_space_sz +
                   variance_space_sz, mean_var_count_grid_desc_m_g.GetElementSize() *
                   sizeof(int32_t), hipMemcpyDeviceToHost);

                                dumpBufferToFile("dump_imm_mean.bin", imm_mean,
                   mean_var_count_grid_desc_m_g.GetElementSize());
                                dumpBufferToFile("dump_imm_var.bin", imm_var,
                   mean_var_count_grid_desc_m_g.GetElementSize());
                                dumpBufferToFile("dump_imm_count.bin", imm_count,
                   mean_var_count_grid_desc_m_g.GetElementSize());

                                delete [] imm_mean;
                                delete [] imm_var;
                                delete [] imm_count;
                */
                avg_time += launch_and_time_kernel(stream_config,
                                                   kern_welford_second_half_batchnorm_forward_final,
                                                   dim3(arg.gridSize),
                                                   dim3(BlockSize),
                                                   0,
                                                   arg.x_grid_desc_m_k,
                                                   arg.y_grid_desc_m_k,
                                                   mean_var_count_grid_desc_m_k,
                                                   arg.scale_bias_grid_desc_m,
                                                   arg.mean_var_grid_desc_m,
                                                   arg.blkGroupSize,
                                                   arg.numBlockTileIteration,
                                                   numMeanVarCountBlockTileIteration,
                                                   arg.epsilon_,
                                                   static_cast<const void*>(arg.p_workspace_),
                                                   arg.p_x_,
                                                   arg.p_scale_,
                                                   arg.p_bias_,
                                                   arg.p_y_,
                                                   arg.updateMovingAverage,
                                                   arg.averageFactor_,
                                                   arg.resultRunningMean_,
                                                   arg.resultRunningVariance_,
                                                   arg.saveMeanInvVariance,
                                                   arg.resultSaveMean_,
                                                   arg.resultSaveInvVariance_);
            }
            else
            {
                using GridwiseBatchNormForwardWithBlockwiseWelford_ =
                    GridwiseBatchNormForwardWithBlockwiseWelford<XDataType,
                                                                 YDataType,
                                                                 AccDataType,
                                                                 ScaleBiasDataType,
                                                                 MeanVarDataType,
                                                                 XYGridDesc_M_K,
                                                                 ScaleBiasMeanVarGridDesc_M,
                                                                 ScaleBiasMeanVarGridDesc_M,
                                                                 BlockSize,
                                                                 MThreadClusterSize,
                                                                 KThreadClusterSize,
                                                                 MThreadSliceSize,
                                                                 KThreadSliceSize,
                                                                 XSrcYDstVectorDim,
                                                                 XSrcVectorSize,
                                                                 YDstVectorSize,
                                                                 ScaleBiasSrcVectorSize,
                                                                 MeanVarSrcDstVectorSize>;

                const auto kern_batchnorm_fwd = kernel_batchnorm_forward_with_blockwise_welford<
                    GridwiseBatchNormForwardWithBlockwiseWelford_,
                    XDataType,
                    YDataType,
                    AccDataType,
                    ScaleBiasDataType,
                    MeanVarDataType,
                    XYGridDesc_M_K,
                    ScaleBiasMeanVarGridDesc_M,
                    ScaleBiasMeanVarGridDesc_M>;

                avg_time +=
                    launch_and_time_kernel(stream_config,
                                           kern_batchnorm_fwd,
                                           dim3(arg.gridSize),
                                           dim3(BlockSize),
                                           0,
                                           arg.x_grid_desc_m_k,
                                           arg.y_grid_desc_m_k,
                                           arg.scale_bias_grid_desc_m,
                                           arg.mean_var_grid_desc_m,
                                           arg.numBlockTileIteration,
                                           arg.epsilon_,
                                           arg.p_x_,
                                           static_cast<const int8_t*>(arg.workspace_initial_count),
                                           arg.p_scale_,
                                           arg.p_bias_,
                                           arg.p_y_,
                                           arg.updateMovingAverage, // true or false
                                           arg.averageFactor_,
                                           arg.resultRunningMean_,
                                           arg.resultRunningVariance_,
                                           arg.saveMeanInvVariance, // true or false
                                           arg.resultSaveMean_,
                                           arg.resultSaveInvVariance_);
            };

            return (avg_time);
        };

        float Run(const BaseArgument* pArg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(pArg), stream_config);
        };
    };

    bool IsSupportedArgument(const BaseArgument* pArg) override
    {
        const Argument* pArg_ = dynamic_cast<const Argument*>(pArg);

        if constexpr(XSrcYDstVectorDim == 0)
        {
            if(pArg_->xStrides_[NumInvariantDim - 1] != 1 ||
               pArg_->yStrides_[NumInvariantDim - 1] != 1)
                return false;

            if(pArg_->xyLengths_[NumInvariantDim - 1] % XSrcVectorSize != 0 ||
               pArg_->xyLengths_[NumInvariantDim - 1] % YDstVectorSize != 0)
                return false;
        }
        else
        {
            if(pArg_->xStrides_[Rank - 1] != 1 || pArg_->yStrides_[Rank - 1] != 1)
                return false;

            if(pArg_->xyLengths_[Rank - 1] % XSrcVectorSize != 0 ||
               pArg_->xyLengths_[Rank - 1] % YDstVectorSize != 0)
                return false;
        };

        if(pArg_->bnScaleBiasStrides_[NumInvariantDim - 1] != 1 && ScaleBiasSrcVectorSize != 1)
            return false;

        if(pArg_->bnScaleBiasMeanVarLengths_[NumInvariantDim - 1] % ScaleBiasSrcVectorSize != 0)
            return false;

        if(pArg_->bnMeanVarStrides_[NumInvariantDim - 1] != 1 && MeanVarSrcDstVectorSize != 1)
            return false;

        if(pArg_->bnScaleBiasMeanVarLengths_[NumInvariantDim - 1] % MeanVarSrcDstVectorSize != 0)
            return false;

        bool is_valid = true;

        static_for<0, NumInvariantDim, 1>{}([&](auto I) {
            if(pArg_->xyLengths_[I] != pArg_->bnScaleBiasMeanVarLengths_[I])
                is_valid = false;
        });

        if(!is_valid)
            return false;

        return true;
    };

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const std::array<index_t, Rank> xyLengths,
        const std::array<index_t, Rank> xStrides,
        const std::array<index_t, Rank> yStrides,
        const std::array<int, NumBatchNormReduceDim> reduceDims,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasMeanVarLengths,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnScaleBiasStrides,
        const std::array<index_t, Rank - NumBatchNormReduceDim> bnMeanVarStrides,
        const void* p_x,
        const void* p_scale,
        const void* p_bias,
        double epsilon,
        void* p_y,
        void* resultSaveMean,
        void* resultSaveInvVariance,
        double averageFactor,
        void* resultRunningMean,
        void* resultRunningVariance) override
    {
        return std::make_unique<Argument>(xyLengths,
                                          xStrides,
                                          yStrides,
                                          reduceDims,
                                          bnScaleBiasMeanVarLengths,
                                          bnScaleBiasStrides,
                                          bnMeanVarStrides,
                                          static_cast<const XDataType*>(p_x),
                                          static_cast<const ScaleBiasDataType*>(p_scale),
                                          static_cast<const ScaleBiasDataType*>(p_bias),
                                          epsilon,
                                          static_cast<YDataType*>(p_y),
                                          static_cast<MeanVarDataType*>(resultSaveMean),
                                          static_cast<MeanVarDataType*>(resultSaveInvVariance),
                                          averageFactor,
                                          static_cast<MeanVarDataType*>(resultRunningMean),
                                          static_cast<MeanVarDataType*>(resultRunningVariance));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchNormImpl<" << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ",";
        str << "XSrcYDstVectorDim_" << XSrcYDstVectorDim  << ",";
        str << "VectorSize_X" << XSrcVectorSize << "_scale_bias_" << ScaleBiasSrcVectorSize << "_mean_var_" << MeanVarSrcDstVectorSize << "_Y" << YDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
