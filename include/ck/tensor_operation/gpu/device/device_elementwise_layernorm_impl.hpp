// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/reduction_operator.hpp"

#include "ck/tensor_operation/gpu/device/device_elementwise_normalization.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_layernorm_welford_variance.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_set_buffer_value.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

// Y = layernorm( A + B, Beta, Gamma )
namespace ck {
template <typename GridwiseElementwiseReduction,
          typename InDataTypePointerTuple,  // Datatype tuple of A & B
          typename CDataType,               // Datatype of A + B
          typename GammaDataType,           // Datatype of Gamma
          typename BetaDataType,            // Datatype of Beta
          typename YDataType,               // Datatype of input Y
          typename AccDataType,             // AccDatatype
          typename ElementwiseOperation,    // Operation of A & B -> Add
          typename AccElementwiseOperation, // Operation Passthrough
          typename InGrid2dDescTuple,       // Descriptor tuple of A & B
          typename GridDesc_M_K,            // Descriptor of A B A+B
          typename GridDesc_K>              // Descriptor of Gamma, Beta
__global__ void kernel_elementwise_layernorm(
    const InGrid2dDescTuple in_grid_2d_desc_tuple,          // Descriptor tuple of A & B
    const GridDesc_M_K c_grid_desc_m_k,                     // Descriptor of C
    const GridDesc_K gamma_grid_desc_k,                     // Descriptor of gamma
    const GridDesc_K beta_grid_desc_k,                      // Descriptor of beta
    const GridDesc_M_K y_grid_desc_m_k,                     // Descriptor of Y
    index_t num_k_block_tile_iteration,                     // arg.numBlockTileIteration_
    AccDataType epsilon,                                    // Datatype of epsilon
    const InDataTypePointerTuple p_in_global_tuple,         // Ptr tuple of input matrixs
    CDataType* const __restrict__ p_c_global,               // Ptr of C
    const GammaDataType* const __restrict__ p_gamma_global, // Ptr of gamma
    const BetaDataType* const __restrict__ p_beta_global,   // Ptr of beta
    YDataType* const __restrict__ p_y_global,               // Ptr of y
    const ElementwiseOperation elementwise_op,              // Operation Add
    const AccElementwiseOperation acc_elementwise_op)       // Operation Passthrough
{
    GridwiseElementwiseReduction::Run(in_grid_2d_desc_tuple,      // Descriptor tuple of A & B
                                      c_grid_desc_m_k,            // Descriptor of C
                                      gamma_grid_desc_k,          // Descriptor of Gamma
                                      beta_grid_desc_k,           // Descriptor of Beta
                                      y_grid_desc_m_k,            // Descriptor of Y
                                      num_k_block_tile_iteration, // arg.numBlockTileIteration_
                                      epsilon,                    // epsilon
                                      p_in_global_tuple,          // Ptr tuple of A & B
                                      p_c_global,                 // Ptr of C
                                      p_gamma_global,             // Ptr of gamma
                                      p_beta_global,              // Ptr of beta
                                      p_y_global,                 // Ptr of Y
                                      elementwise_op,             // Add
                                      acc_elementwise_op);        // Passthrough
};
} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// Y = LayerNorm(A + B, Beta, Gamma)
template <typename InDataTypeTuple, // Datatype of A & B
          typename CDataType,       // Datatype of C = A + B
          typename GammaDataType,   //
          typename BetaDataType,
          typename AccDataType,
          typename YDataType,
          typename ElementwiseOperation,
          typename AccElementwiseOperation,
          index_t Rank,               //
          index_t NumReduceDim,       //
          index_t BlockSize,          //
          index_t MThreadClusterSize, // Num of threads in a block on M direction
          index_t KThreadClusterSize, // Num of threads in a block on N direction
          index_t MThreadSliceSize,   // Each thread calculate rows
          index_t KThreadSliceSize,   // Each thread calculate columns
          index_t XYSrcVectorDim,     // Dimension to do reduce
          index_t XSrcVectorSize,     // Size to fetch source x
          index_t GammaSrcVectorSize, // Size to fetch source gamma
          index_t BetaSrcVectorSize,  // Size to fetch source beta
          index_t YDstVectorSize>     // Size to write destination Y
struct DeviceElementwiseLayernormImpl : public DeviceElementwiseLayernorm<InDataTypeTuple,
                                                                          CDataType,
                                                                          GammaDataType,
                                                                          BetaDataType,
                                                                          AccDataType,
                                                                          YDataType,
                                                                          ElementwiseOperation,
                                                                          AccElementwiseOperation,
                                                                          Rank,
                                                                          NumReduceDim>
{
    static constexpr int NumInput = InDataTypeTuple::Size();

    static_assert(
        (KThreadSliceSize % GammaSrcVectorSize == 0),
        "Invalid thread slice sizes and/or gamma vector sizes configuration, please check!");

    static_assert(
        (KThreadSliceSize % BetaSrcVectorSize == 0),
        "Invalid thread slice sizes and/or beta vector sizes configuration, please check!");

    using PassThrough = tensor_operation::element_wise::PassThrough;

    static constexpr index_t M_BlockTileSize =
        MThreadClusterSize * MThreadSliceSize; // num of rows calculated in a block
    static constexpr index_t K_BlockTileSize =
        KThreadClusterSize * KThreadSliceSize; // num of columns calculated in a block

    static auto GenerateInDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;

                return static_cast<const DataType*>(nullptr);
            },
            Number<NumInput>{});
    };

    using InDataTypePointerTuple = decltype(GenerateInDataTypePointerTuple());

    static auto MakeSrc2dDescriptor(const std::vector<index_t>& inLengths,
                                    const std::vector<index_t>& inStrides,
                                    int blkGroupSize,
                                    int numBlockTileIteration)
    {
        constexpr index_t NumInvariantDim  = Rank - NumReduceDim;
        static constexpr index_t numSrcDim = Rank;
        static constexpr bool reduceAllDim = (NumInvariantDim == 0);

        const auto tupleSrcLengths = make_tuple_from_array(inLengths, Number<numSrcDim>{});
        const auto tupleSrcStrides = make_tuple_from_array(inStrides, Number<numSrcDim>{});

        const auto inDesc = make_naive_tensor_descriptor(tupleSrcLengths, tupleSrcStrides);

        const auto in_grid_desc_m_k = [&]() {
            if constexpr(reduceAllDim)
            {
                const auto one_dim_inDesc = transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(tupleSrcLengths)),
                    make_tuple(typename arithmetic_sequence_gen<0, numSrcDim, 1>::type{}),
                    make_tuple(Sequence<0>{}));

                return transform_tensor_descriptor(one_dim_inDesc,
                                                   make_tuple(make_unmerge_transform(make_tuple(
                                                       1, one_dim_inDesc.GetLength(Number<0>{})))),
                                                   make_tuple(Sequence<0>{}),
                                                   make_tuple(Sequence<0, 1>{}));
            }
            else
            {
                using InvariantDims = typename arithmetic_sequence_gen<0, NumInvariantDim, 1>::type;
                using ReduceDims = typename arithmetic_sequence_gen<NumInvariantDim, Rank, 1>::type;

                const auto reduceDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, ReduceDims{});
                const auto invariantDimLengths =
                    make_tuple_from_array_and_index_seq(inLengths, InvariantDims{});

                return transform_tensor_descriptor(
                    inDesc,
                    make_tuple(make_merge_transform(invariantDimLengths),
                               make_merge_transform(reduceDimLengths)),
                    make_tuple(InvariantDims{}, ReduceDims{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }();

        const auto invariantLength = in_grid_desc_m_k.GetLength(Number<0>{});
        const auto reduceLength    = in_grid_desc_m_k.GetLength(Number<1>{});

        const int reduceSizePerBlock = K_BlockTileSize * numBlockTileIteration;
        const auto inPad_M =
            math::integer_least_multiple(invariantLength, M_BlockTileSize) - invariantLength;
        const auto inPad_K = reduceSizePerBlock * blkGroupSize - reduceLength;

        auto in_grid_desc_m_k_padded = transform_tensor_descriptor(
            in_grid_desc_m_k,
            make_tuple(make_right_pad_transform(invariantLength, inPad_M),
                       make_right_pad_transform(reduceLength, inPad_K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return (in_grid_desc_m_k_padded);
    };

    static auto MakeAffine1dDescriptor(const std::vector<index_t>& Lengths,
                                       const std::vector<index_t>& Strides,
                                       int blkGroupSize,
                                       int numBlockTileIteration)
    {
        const auto tupleLengths = make_tuple_from_array(Lengths, Number<NumReduceDim>{});
        const auto tupleStrides = make_tuple_from_array(Strides, Number<NumReduceDim>{});

        auto desc = make_naive_tensor_descriptor(tupleLengths, tupleStrides);

        auto grid_desc_k = transform_tensor_descriptor(
            desc,
            make_tuple(make_merge_transform(tupleLengths)),
            make_tuple(typename arithmetic_sequence_gen<0, NumReduceDim, 1>::type{}),
            make_tuple(Sequence<0>{}));

        const auto reduceTotalLength = grid_desc_k.GetLength(Number<0>{});

        const int reduceSizePerBlock = K_BlockTileSize * numBlockTileIteration;

        const auto Pad_K = reduceSizePerBlock * blkGroupSize - reduceTotalLength;

        auto grid_desc_k_padded = transform_tensor_descriptor(
            grid_desc_k,
            make_tuple(make_right_pad_transform(reduceTotalLength, Pad_K)),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0>{}));

        return (grid_desc_k_padded);
    };

    template <index_t TupleSize>
    static auto GenerateSrcGrid2dDescTuple(Number<TupleSize>)
    {
        return generate_tuple([&](auto) { return MakeSrc2dDescriptor({1}, {1}, 1, 1); },
                              Number<TupleSize>{});
    };

    using InGrid2dDescTuple = decltype(GenerateSrcGrid2dDescTuple(Number<NumInput>{}));

    using GridDesc_M_K = decltype(MakeSrc2dDescriptor({1}, {1}, 1, 1));
    using GridDesc_K   = decltype(MakeAffine1dDescriptor({1}, {1}, 1, 1));

    using GridwiseReduceLayernormGeneric =
        GridwiseElementwiseLayernormWelfordVariance_mk_to_mk<InDataTypePointerTuple,
                                                             CDataType,
                                                             GammaDataType,
                                                             BetaDataType,
                                                             YDataType,
                                                             AccDataType,
                                                             ElementwiseOperation,
                                                             AccElementwiseOperation,
                                                             InGrid2dDescTuple,
                                                             GridDesc_M_K,
                                                             GridDesc_K,
                                                             BlockSize,
                                                             MThreadClusterSize,
                                                             KThreadClusterSize,
                                                             MThreadSliceSize,
                                                             KThreadSliceSize,
                                                             XYSrcVectorDim,
                                                             XSrcVectorSize,
                                                             GammaSrcVectorSize,
                                                             BetaSrcVectorSize,
                                                             XYSrcVectorDim,
                                                             YDstVectorSize,
                                                             false>;

    using GridwiseReduceLayernormSweepOnce =
        GridwiseElementwiseLayernormWelfordVariance_mk_to_mk<InDataTypePointerTuple,
                                                             CDataType,
                                                             GammaDataType,
                                                             BetaDataType,
                                                             YDataType,
                                                             AccDataType,
                                                             ElementwiseOperation,
                                                             AccElementwiseOperation,
                                                             InGrid2dDescTuple,
                                                             GridDesc_M_K,
                                                             GridDesc_K,
                                                             BlockSize,
                                                             MThreadClusterSize,
                                                             KThreadClusterSize,
                                                             MThreadSliceSize,
                                                             KThreadSliceSize,
                                                             XYSrcVectorDim,
                                                             XSrcVectorSize,
                                                             GammaSrcVectorSize,
                                                             BetaSrcVectorSize,
                                                             XYSrcVectorDim,
                                                             YDstVectorSize,
                                                             true>;

    struct Argument : public BaseArgument
    {
        Argument(const std::vector<index_t> lengths,
                 const std::array<std::vector<index_t>, NumInput> inStridesArray,
                 const std::vector<index_t> cStrides,
                 const std::vector<index_t> gammaStrides,
                 const std::vector<index_t> betaStrides,
                 const std::vector<index_t> yStrides,
                 const std::vector<index_t> reduceDims,
                 ElementwiseOperation elementwise_op,
                 AccElementwiseOperation acc_elementwise_op,
                 AccDataType epsilon,
                 const std::array<const void*, NumInput> in_dev_buffers,
                 CDataType* p_c,
                 const GammaDataType* p_gamma,
                 const BetaDataType* p_beta,
                 YDataType* p_y)
            : epsilon_(epsilon),
              p_c_(p_c),
              p_gamma_(p_gamma),
              p_beta_(p_beta),
              p_y_(p_y),
              gammaStrides_(gammaStrides),
              betaStrides_(betaStrides),
              elementwise_op_(elementwise_op),
              acc_elementwise_op_(acc_elementwise_op)
        {
            Lengths_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(lengths, reduceDims);
            for(int i = 0; i < NumInput; i++)
            {
                inStridesArray_[i] =
                    shuffle_tensor_dimensions<Rank, NumReduceDim>(inStridesArray[i], reduceDims);
            }
            cStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(cStrides, reduceDims);
            yStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(yStrides, reduceDims);

            in_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;
                    return static_cast<const DataType*>(in_dev_buffers[I.value]);
                },
                Number<NumInput>{});

            long_index_t invariant_total_length;
            long_index_t reduce_total_length;

            std::tie(invariant_total_length, reduce_total_length) =
                get_2d_lengths<Rank, NumReduceDim>(Lengths_);

            blkGroupSize_          = 1;
            numBlockTileIteration_ = (reduce_total_length + K_BlockTileSize - 1) / K_BlockTileSize;

            gridSize_ = math::integer_least_multiple(invariant_total_length, M_BlockTileSize) /
                        M_BlockTileSize * blkGroupSize_;

            reduceLengths_.resize(NumReduceDim);

            for(int i = 0; i < NumReduceDim; ++i)
            {
                reduceLengths_[i] = lengths[reduceDims[i]];
            }
        }

        AccDataType epsilon_;

        InDataTypePointerTuple in_dev_buffers_;
        CDataType* p_c_;
        const GammaDataType* p_gamma_;
        const BetaDataType* p_beta_;
        YDataType* p_y_;

        std::vector<index_t> Lengths_;
        std::array<std::vector<index_t>, NumInput> inStridesArray_;
        std::vector<index_t> cStrides_;
        std::vector<index_t> reduceLengths_;
        std::vector<index_t> gammaStrides_;
        std::vector<index_t> betaStrides_;
        std::vector<index_t> yStrides_;

        ElementwiseOperation elementwise_op_;
        AccElementwiseOperation acc_elementwise_op_;

        int blkGroupSize_;
        int numBlockTileIteration_;
        size_t gridSize_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const InGrid2dDescTuple in_grid_2d_desc_tuple = generate_tuple(
                [&](auto I) {
                    return MakeSrc2dDescriptor(arg.Lengths_,
                                               arg.inStridesArray_[I.value],
                                               arg.blkGroupSize_,
                                               arg.numBlockTileIteration_);
                },
                Number<NumInput>{});

            const auto c_grid_desc_m_k = MakeSrc2dDescriptor(
                arg.Lengths_, arg.cStrides_, arg.blkGroupSize_, arg.numBlockTileIteration_);

            const auto gamma_grid_desc_k = MakeAffine1dDescriptor(arg.reduceLengths_,
                                                                  arg.gammaStrides_,
                                                                  arg.blkGroupSize_,
                                                                  arg.numBlockTileIteration_);
            const auto beta_grid_desc_k  = MakeAffine1dDescriptor(arg.reduceLengths_,
                                                                 arg.betaStrides_,
                                                                 arg.blkGroupSize_,
                                                                 arg.numBlockTileIteration_);
            const auto y_grid_desc_m_k   = MakeSrc2dDescriptor(
                arg.Lengths_, arg.yStrides_, arg.blkGroupSize_, arg.numBlockTileIteration_);

            bool sweep_once =
                c_grid_desc_m_k.GetLength(Number<1>{}) <= KThreadClusterSize * KThreadSliceSize;

            const auto kernel_main =
                sweep_once ? kernel_elementwise_layernorm<GridwiseReduceLayernormSweepOnce,
                                                          InDataTypePointerTuple,
                                                          CDataType,
                                                          GammaDataType,
                                                          BetaDataType,
                                                          YDataType,
                                                          AccDataType,
                                                          ElementwiseOperation,
                                                          AccElementwiseOperation,
                                                          InGrid2dDescTuple,
                                                          GridDesc_M_K,
                                                          GridDesc_K>
                           : kernel_elementwise_layernorm<GridwiseReduceLayernormGeneric,
                                                          InDataTypePointerTuple,
                                                          CDataType,
                                                          GammaDataType,
                                                          BetaDataType,
                                                          YDataType,
                                                          AccDataType,
                                                          ElementwiseOperation,
                                                          AccElementwiseOperation,
                                                          InGrid2dDescTuple,
                                                          GridDesc_M_K,
                                                          GridDesc_K>;

            float avg_time = 0;
            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize_),
                                               dim3(BlockSize),
                                               0,
                                               in_grid_2d_desc_tuple,
                                               c_grid_desc_m_k,
                                               gamma_grid_desc_k,
                                               beta_grid_desc_k,
                                               y_grid_desc_m_k,
                                               arg.numBlockTileIteration_,
                                               arg.epsilon_,
                                               arg.in_dev_buffers_,
                                               arg.p_c_,
                                               arg.p_gamma_,
                                               arg.p_beta_,
                                               arg.p_y_,
                                               arg.elementwise_op_,
                                               arg.acc_elementwise_op_);

            return (avg_time);
        };

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        };
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* p_arg_ = dynamic_cast<const Argument*>(p_arg);

        constexpr index_t NumInvariantDim = Rank - NumReduceDim;

        if constexpr(XYSrcVectorDim == 0)
        {
            if constexpr(NumInvariantDim == 0)
            {
                return false;
            }
            else
            {
                for(int i = 0; i < NumInput; i++)
                {
                    if(p_arg_->inStridesArray_[i][NumInvariantDim - 1] != 1)
                        return false;
                }

                if(p_arg_->inStridesArray_[0][NumInvariantDim - 1] != 1 &&
                   p_arg_->inStridesArray_[1][NumInvariantDim - 1] != 1)
                    return false;

                if(p_arg_->invariant_lowest_length % XSrcVectorSize != 0)
                    return false;
            };
        }
        else
        {
            for(int i = 0; i < NumInput; i++)
            {
                if(p_arg_->inStridesArray_[i][Rank - 1] != 1)
                    return false;
            }

            if(p_arg_->Lengths_[Rank - 1] % XSrcVectorSize != 0)
                return false;
        };

        if(p_arg_->Lengths_[Rank - 1] % YDstVectorSize != 0)
        {
            return false;
        }

        if(p_arg_->gammaStrides_.size() != NumReduceDim ||
           p_arg_->betaStrides_.size() != NumReduceDim)
            return false;

        auto IsScalarPerVectorValid = [](bool isLastDimensionCoalesced, int scalarPerVector) {
            bool ret = true;

            if(!isLastDimensionCoalesced)
                ret = scalarPerVector == 1;
            else
                ret = KThreadSliceSize % scalarPerVector == 0;

            return ret;
        };

        if(!IsScalarPerVectorValid(p_arg_->gammaStrides_.back() == 1, GammaSrcVectorSize))
            return false;

        if(!IsScalarPerVectorValid(p_arg_->betaStrides_.back() == 1, BetaSrcVectorSize))
            return false;

        return true;
    };

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::vector<index_t> lengths,
                        const std::array<std::vector<index_t>, NumInput> inStridesArray,
                        const std::vector<index_t> cStrides,
                        const std::vector<index_t> gammaStrides,
                        const std::vector<index_t> betaStrides,
                        const std::vector<index_t> yStrides,
                        const std::vector<index_t> reduceDims,
                        AccDataType epsilon,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        void* p_c,
                        const void* p_gamma,
                        const void* p_beta,
                        void* p_y,
                        ElementwiseOperation elementwise_op,
                        AccElementwiseOperation acc_elementwise_op) override
    {
        return std::make_unique<Argument>(lengths,
                                          inStridesArray,
                                          cStrides,
                                          gammaStrides,
                                          betaStrides,
                                          yStrides,
                                          reduceDims,
                                          elementwise_op,
                                          acc_elementwise_op,
                                          epsilon,
                                          in_dev_buffers,
                                          static_cast<CDataType*>(p_c),
                                          static_cast<const GammaDataType*>(p_gamma),
                                          static_cast<const BetaDataType*>(p_beta),
                                          static_cast<YDataType*>(p_y));
    };

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceAddLayernormImpl<" << BlockSize << ",";
        str << "M_C" << MThreadClusterSize << "_S" << MThreadSliceSize << ",";
        str << "K_C" << KThreadClusterSize << "_S" << KThreadSliceSize << ",";
        str << "XYSrcVectorDim_" << XYSrcVectorDim  << ",";
        str << "VectorSize_X" << XSrcVectorSize << "_Gamma" << GammaSrcVectorSize << "_Beta" << BetaSrcVectorSize << "_Y" << YDstVectorSize << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
