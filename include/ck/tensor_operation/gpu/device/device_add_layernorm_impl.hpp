// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/reduction_operator.hpp"
#include "ck/tensor_operation/gpu/device/device_add_normalization.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce.hpp"
#include "ck/tensor_operation/gpu/device/device_reduce_common.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_add_layernorm_welford_variance.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_set_buffer_value.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
// Y = layernorm( A + B, Beta, Gamma )
namespace ck {
template <typename GridwiseaddReduction,
          typename ADataType,               // Datatype of input A
          typename BDataType,               // Datatype of input B
          typename CDataType,               // Datatype of A + B
          typename GammaDataType,           // Datatype of Gamma
          typename BetaDataType,            // Datatype of Beta
          typename YDataType,               // Datatype of input Y
          typename AccDataType,             // AccDatatype
          typename ElementwiseOperation,    // Operation of A & B -> Add
          typename AccElementwiseOperation, // Operation Passthrough
          typename GridDesc_M_K,            // Descriptor of A B A+B
          typename GridDesc_K>              // Descriptor of Gamma, Beta
__global__ void
kernel_add_layernorm(const GridDesc_M_K a_grid_desc_m_k,             // Descriptor of A
                     const GridDesc_M_K b_grid_desc_m_k,             // Descriptor of B
                     const GridDesc_M_K c_grid_desc_m_k,             // Descriptor of C
                     const GridDesc_K gamma_grid_desc_k,             // Descriptor of gamma
                     const GridDesc_K beta_grid_desc_k,              // Descriptor of beta
                     const GridDesc_M_K y_grid_desc_m_k,             // Descriptor of Y
                     index_t num_k_block_tile_iteration,             // arg.numBlockTileIteration_
                     AccDataType epsilon,                            // Datatype of epsilon
                     const ADataType* const __restrict__ p_a_global, // Ptr of A
                     const BDataType* const __restrict__ p_b_global, // Ptr of B
                     CDataType* const __restrict__ p_c_global,       // Ptr of C
                     const GammaDataType* const __restrict__ p_gamma_global, // Ptr of gamma
                     const BetaDataType* const __restrict__ p_beta_global,   // Ptr of beta
                     YDataType* const __restrict__ p_y_global,               // Ptr of y
                     const ElementwiseOperation elementwise_op,              // Operation Add
                     const AccElementwiseOperation acc_elementwise_op) // Operation Passthrough
{
    GridwiseaddReduction::Run(a_grid_desc_m_k,            // Descriptor of A
                              b_grid_desc_m_k,            // Descriptor of B
                              c_grid_desc_m_k,            // Descriptor of C
                              gamma_grid_desc_k,          // Descriptor of Gamma
                              beta_grid_desc_k,           // Descriptor of Beta
                              y_grid_desc_m_k,            // Descriptor of Y
                              num_k_block_tile_iteration, // arg.numBlockTileIteration_
                              epsilon,                    //
                              p_a_global,                 // Ptr of A
                              p_b_global,                 // Ptr of B
                              p_c_global,                 // Ptr of C
                              p_gamma_global,             // Ptr of gamma
                              p_beta_global,              // Ptr of beta
                              p_y_global,                 // Ptr of Y
                              elementwise_op,             //
                              acc_elementwise_op);        //
};
} // namespace ck

namespace ck {
namespace tensor_operation {
namespace device {

// Y = LayerNorm(A + B, Beta, Gamma)
template <typename ADataType, // Datatype of input A
          typename BDataType, // Datatype of input B
          typename CDataType, // Datatype of C = A + B
          typename GammaDataType,
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
struct DeviceAddLayernormImpl : public DeviceAddLayernorm<ADataType,
                                                          BDataType,
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

    using GridDesc_M_K = decltype(MakeSrc2dDescriptor({1}, {1}, 1, 1));
    using GridDesc_K   = decltype(MakeAffine1dDescriptor({1}, {1}, 1, 1));

    using GridwiseReduceLayernormGeneric =
        GridwiseAddLayernormWelfordVariance_mk_to_mk<ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     GammaDataType,
                                                     BetaDataType,
                                                     YDataType,
                                                     AccDataType,
                                                     ElementwiseOperation,
                                                     AccElementwiseOperation,
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
        GridwiseAddLayernormWelfordVariance_mk_to_mk<ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     GammaDataType,
                                                     BetaDataType,
                                                     YDataType,
                                                     AccDataType,
                                                     ElementwiseOperation,
                                                     AccElementwiseOperation,
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
                 const std::vector<index_t> aStrides,
                 const std::vector<index_t> bStrides,
                 const std::vector<index_t> cStrides,
                 const std::vector<index_t> gammaStrides,
                 const std::vector<index_t> betaStrides,
                 const std::vector<index_t> yStrides,
                 const std::vector<index_t> reduceDims,
                 ElementwiseOperation elementwise_op,
                 AccElementwiseOperation acc_elementwise_op,
                 AccDataType epsilon,
                 const ADataType* p_a,
                 const BDataType* p_b,
                 CDataType* p_c,
                 const GammaDataType* p_gamma,
                 const BetaDataType* p_beta,
                 YDataType* p_y)
            : epsilon_(epsilon),
              p_a_(p_a),
              p_b_(p_b),
              p_c_(p_c),
              p_gamma_(p_gamma),
              p_beta_(p_beta),
              p_y_(p_y),
              gammaStrides_(gammaStrides),
              betaStrides_(betaStrides),
              elementwise_op_(elementwise_op),
              acc_elementwise_op_(acc_elementwise_op)
        {
            Lengths_  = shuffle_tensor_dimensions<Rank, NumReduceDim>(lengths, reduceDims);
            aStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(aStrides, reduceDims);
            bStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(bStrides, reduceDims);
            cStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(cStrides, reduceDims);
            yStrides_ = shuffle_tensor_dimensions<Rank, NumReduceDim>(yStrides, reduceDims);

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

        const ADataType* p_a_;
        const BDataType* p_b_;
        CDataType* p_c_;
        const GammaDataType* p_gamma_;
        const BetaDataType* p_beta_;
        YDataType* p_y_;

        std::vector<index_t> Lengths_;
        std::vector<index_t> aStrides_;
        std::vector<index_t> bStrides_;
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
            const auto a_grid_desc_m_k = MakeSrc2dDescriptor(
                arg.Lengths_, arg.aStrides_, arg.blkGroupSize_, arg.numBlockTileIteration_);

            const auto b_grid_desc_m_k = MakeSrc2dDescriptor(
                arg.Lengths_, arg.bStrides_, arg.blkGroupSize_, arg.numBlockTileIteration_);

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
                a_grid_desc_m_k.GetLength(Number<1>{}) <= KThreadClusterSize * KThreadSliceSize;

            const auto kernel_main = sweep_once
                                         ? kernel_add_layernorm<GridwiseReduceLayernormSweepOnce,
                                                                ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                GammaDataType,
                                                                BetaDataType,
                                                                YDataType,
                                                                AccDataType,
                                                                ElementwiseOperation,
                                                                AccElementwiseOperation,
                                                                GridDesc_M_K,
                                                                GridDesc_K>
                                         : kernel_add_layernorm<GridwiseReduceLayernormGeneric,
                                                                ADataType,
                                                                BDataType,
                                                                CDataType,
                                                                GammaDataType,
                                                                BetaDataType,
                                                                YDataType,
                                                                AccDataType,
                                                                ElementwiseOperation,
                                                                AccElementwiseOperation,
                                                                GridDesc_M_K,
                                                                GridDesc_K>;

            float avg_time = 0;
            avg_time += launch_and_time_kernel(stream_config,
                                               kernel_main,
                                               dim3(arg.gridSize_),
                                               dim3(BlockSize),
                                               0,
                                               a_grid_desc_m_k,
                                               b_grid_desc_m_k,
                                               c_grid_desc_m_k,
                                               gamma_grid_desc_k,
                                               beta_grid_desc_k,
                                               y_grid_desc_m_k,
                                               arg.numBlockTileIteration_,
                                               arg.epsilon_,
                                               arg.p_a_,
                                               arg.p_b_,
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
                if(p_arg_->aStrides_[NumInvariantDim - 1] != 1)
                    return false;

                if(p_arg_->invariant_lowest_length % XSrcVectorSize != 0)
                    return false;
            };
        }
        else
        {
            if(p_arg_->aStrides_[Rank - 1] != 1)
                return false;

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
                        const std::vector<index_t> aStrides,
                        const std::vector<index_t> bStrides,
                        const std::vector<index_t> cStrides,
                        const std::vector<index_t> gammaStrides,
                        const std::vector<index_t> betaStrides,
                        const std::vector<index_t> yStrides,
                        const std::vector<index_t> reduceDims,
                        AccDataType epsilon,
                        const void* p_a,
                        const void* p_b,
                        void* p_c,
                        const void* p_gamma,
                        const void* p_beta,
                        void* p_y,
                        ElementwiseOperation elementwise_op,
                        AccElementwiseOperation acc_elementwise_op) override
    {
        return std::make_unique<Argument>(lengths,
                                          aStrides,
                                          bStrides,
                                          cStrides,
                                          gammaStrides,
                                          betaStrides,
                                          yStrides,
                                          reduceDims,
                                          elementwise_op,
                                          acc_elementwise_op,
                                          epsilon,
                                          static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
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
