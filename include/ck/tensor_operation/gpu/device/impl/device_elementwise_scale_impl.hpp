// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/math.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise_scale.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_elementwise_1d_scale.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/stream_utility.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <index_t BlockSize,
          index_t M0PerBlock,
          index_t M1PerBlock,
          typename InDataTypeTuple,
          typename OutDataTypeTuple,
          typename ElementwiseOperation,
          typename UnaryOperation,
          typename Scale,
          index_t NumDim,
          index_t M0PerThread,
          index_t M1PerThread,
          typename ThreadClusterArrangeOrder,
          typename InScalarPerVectorSeq,
          typename OutScalarPerVectorSeq>
struct DeviceElementwiseImpl : public DeviceElementwise<InDataTypeTuple,
                                                        OutDataTypeTuple,
                                                        ElementwiseOperation,
                                                        UnaryOperation,
                                                        Scale,
                                                        NumDim>
{
    static constexpr int NumInput  = InDataTypeTuple::Size();
    static constexpr int NumOutput = OutDataTypeTuple::Size();
    // static constexpr index_t BlockSize = 128;
    // static constexpr index_t threadx1 = 8;
    // static constexpr index_t threadx0 = 16;

    static_assert(NumInput == InScalarPerVectorSeq::Size() &&
                      NumOutput == OutScalarPerVectorSeq::Size(),
                  "Tuple size is inconsistent with the number of in/out!");

    static auto GenerateInDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;

                return static_cast<const DataType*>(nullptr);
            },
            Number<NumInput>{});
    };

    static auto GenerateOutDataTypePointerTuple()
    {
        return generate_tuple(
            [&](auto I) {
                using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;

                return static_cast<DataType*>(nullptr);
            },
            Number<NumOutput>{});
    };

    using InDataTypePointerTuple  = decltype(GenerateInDataTypePointerTuple());
    using OutDataTypePointerTuple = decltype(GenerateOutDataTypePointerTuple());

    template <typename InOutDescriptor>
    static auto PadInputOutputDescriptor(const InOutDescriptor &desc)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};

        const auto M0 = desc.GetLength(I0);
        const auto M1 = desc.GetLength(I1);
        const auto pad_M0 = math::integer_divide_ceil(M0, M0PerThread) * M0PerThread - M0;
        const auto pad_M1 = math::integer_divide_ceil(M1, M1PerThread) * M1PerThread - M1;

        const auto padded_desc =
            transform_tensor_descriptor(desc,
                                        make_tuple(
                                                   make_right_pad_transform(M0, pad_M0),
                                                   make_right_pad_transform(M1, pad_M1)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}));

        return padded_desc;
    }

    static index_t GetMostContinousDim(const std::array<index_t, NumDim>& strides) {
        index_t most_continous_dim = NumDim - 1;
        index_t most_continous_dim_stride = strides[most_continous_dim];
        for (index_t dim = 0; dim < NumDim; dim++) {
            if(strides[dim] < most_continous_dim_stride) {
                most_continous_dim_stride = strides[dim];
                most_continous_dim = dim;
            }
        }
        return most_continous_dim;
    }

    static auto GenerateBatchDimsSizesTuple(const std::array<index_t, NumDim>& lengths, const index_t M0_dim, const index_t M1_dim) {
        std::array<index_t, NumDim - 1> batch_dims;
        index_t batch_dim = 0;
        for (index_t i = 0; i < NumDim; i++) {
            if (i != M0_dim && i != M1_dim) {
                batch_dims[batch_dim] = lengths[i];
                batch_dim++;
            }
        }
        batch_dims[NumDim - 2] = 1;
        return generate_tuple([&](auto I) { return batch_dims[I]; }, Number<NumDim - 1>{});
    }

    static auto GenerateBatchDimsSizesTuple(const std::array<index_t, NumDim>& lengths, const index_t M1_dim) {
        std::array<index_t, NumDim - 1> batch_dims;
        index_t batch_dim = 0;
        for (index_t i = 0; i < NumDim; i++) {
            if (i != M1_dim) {
                batch_dims[batch_dim] = lengths[i];
                batch_dim++;
            }
        }
        return generate_tuple([&](auto I) { return batch_dims[I]; }, Number<NumDim - 1>{});
    }

    static auto MakeInputOutputDescriptor(const std::array<index_t, NumDim>& lengths,
                                 const std::array<index_t, NumDim>& in_strides,
                                 const std::array<index_t, NumDim>& out_strides,
                                 const std::array<index_t, NumDim>& desc_strides)
{
        const auto M0_dim = GetMostContinousDim(out_strides);
        const auto M1_dim = GetMostContinousDim(in_strides);

        const auto M0 = M0_dim == M1_dim ? 1 : lengths[M0_dim];
        const auto M1 = lengths[M1_dim];
        const auto M0_stride = M0_dim == M1_dim ? 1 : desc_strides[M0_dim];
        const auto M1_stride = desc_strides[M1_dim];

        const auto batch_dims_lenghts = M0_dim == M1_dim ? 
                    GenerateBatchDimsSizesTuple(lengths, M1_dim) :
                    GenerateBatchDimsSizesTuple(lengths, M0_dim, M1_dim);
        const auto batch_dims_strides = M0_dim == M1_dim ? 
         GenerateBatchDimsSizesTuple(desc_strides, M1_dim) :
          GenerateBatchDimsSizesTuple(desc_strides, M0_dim, M1_dim);

        const auto desc = make_naive_tensor_descriptor(concat_tuple(batch_dims_lenghts, make_tuple(M0), make_tuple(M1)), concat_tuple(batch_dims_strides, make_tuple(M0_stride), make_tuple(M1_stride)));

        const auto transforms = make_tuple(
            make_merge_transform(concat_tuple(batch_dims_lenghts, make_tuple(M0))),
            make_pass_through_transform(M1)
        );

        using BatchElemsSequence = typename arithmetic_sequence_gen<0, decltype(batch_dims_lenghts)::Size() + 1, 1>::type;
        const auto lower_dims = make_tuple(BatchElemsSequence{}, Sequence<NumDim>{});
        const auto upper_dims = make_tuple(Sequence<1>{}, Sequence<0>{});

        // desc: (merged_dims, b_vector_dim, a_vector_dim)
        auto merged_desc = transform_tensor_descriptor(desc, transforms, lower_dims, upper_dims);
        return PadInputOutputDescriptor(merged_desc);
    }

    template<index_t NumTensors>
    static auto GenerateInOutGridDescTuple()
    {
        std::array<index_t, NumDim> ones;
        for (index_t d = 0; d < NumDim; d++) {
            ones[d] = 1;
        }

        return generate_tuple(
            [&](auto) {
                return MakeInputOutputDescriptor(ones, ones, ones, ones);
            },
            Number<NumTensors>{});
    };

    using InGridDescTuple  = decltype(GenerateInOutGridDescTuple<NumInput>());
    using OutGridDescTuple = decltype(GenerateInOutGridDescTuple<NumOutput>());

    using Block2TileMap = BlockToCTileMap_M00_N0_M01Adapt<M0PerThread * M0PerBlock, M1PerThread * M1PerBlock>;

    using GridwiseElementwise = GridwiseElementwise_1D<InGridDescTuple,
                                                       OutGridDescTuple,
                                                       InDataTypePointerTuple,
                                                       OutDataTypePointerTuple,
                                                       Block2TileMap,
                                                       ElementwiseOperation,
                                                       UnaryOperation,
                                                       Scale,
                                                       M0PerThread,
                                                       M1PerThread,
                                                       ThreadClusterArrangeOrder,
                                                       InScalarPerVectorSeq,
                                                       OutScalarPerVectorSeq,
                                                       BlockSize,
                                                       M0PerBlock,
                                                       M1PerBlock,
                                                       false>;

    using GridwiseElementwiseSameInOutVectorDim = GridwiseElementwise_1D<InGridDescTuple,
                                                       OutGridDescTuple,
                                                       InDataTypePointerTuple,
                                                       OutDataTypePointerTuple,
                                                       Block2TileMap,
                                                       ElementwiseOperation,
                                                       UnaryOperation,
                                                       Scale,
                                                       M0PerThread,
                                                       M1PerThread,
                                                       ThreadClusterArrangeOrder,
                                                       InScalarPerVectorSeq,
                                                       OutScalarPerVectorSeq,
                                                       BlockSize,
                                                       M0PerBlock,
                                                       M1PerBlock,
                                                       true>;
    struct Argument : public BaseArgument
    {
        Argument(const std::array<index_t, NumDim> lengths,
                 const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                 const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                 const std::array<const void*, NumInput> in_dev_buffers,
                 const std::array<void*, NumOutput> out_dev_buffers,
                 ElementwiseOperation elementwise_op,
                 UnaryOperation unary_op,
                 Scale scale_op)

            : lengths_(lengths),
              inStridesArray_(inStridesArray),
              outStridesArray_(outStridesArray),
              elementwise_op_(elementwise_op),
              unary_op_(unary_op),
              scale_op_(scale_op)
        {
            in_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(InDataTypeTuple{}[I])>;
                    return static_cast<const DataType*>(in_dev_buffers[I.value]);
                },
                Number<NumInput>{});

            out_dev_buffers_ = generate_tuple(
                [&](auto I) {
                    using DataType = remove_cvref_t<decltype(OutDataTypeTuple{}[I])>;
                    return static_cast<DataType*>(out_dev_buffers[I.value]);
                },
                Number<NumOutput>{});
        }

        InDataTypePointerTuple in_dev_buffers_;
        OutDataTypePointerTuple out_dev_buffers_;

        std::array<index_t, NumDim> lengths_;
        std::array<std::array<index_t, NumDim>, NumInput> inStridesArray_;
        std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray_;

        ElementwiseOperation elementwise_op_;
        UnaryOperation unary_op_;
        Scale scale_op_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            auto in_grid_desc_tuple = generate_tuple(
                [&](auto) {
                    return MakeInputOutputDescriptor(
                        arg.lengths_, arg.inStridesArray_[0], arg.outStridesArray_[0], arg.inStridesArray_[0]);
                },
                Number<NumInput>{});

            auto out_grid_desc_tuple = generate_tuple(
                [&](auto) {
                    return MakeInputOutputDescriptor(
                        arg.lengths_, arg.inStridesArray_[0], arg.outStridesArray_[0], arg.outStridesArray_[0]);
                },
                Number<NumOutput>{});


            const index_t batch_size = in_grid_desc_tuple.At(Number<0>{}).GetLength(Number<0>{});
            const index_t M0 = in_grid_desc_tuple.At(Number<0>{}).GetLength(Number<0>{});
            const index_t M1 = in_grid_desc_tuple.At(Number<0>{}).GetLength(Number<1>{});
            const auto block_2_tile_map =
                Block2TileMap(M0, M1);

            const index_t grid_size =
                block_2_tile_map.CalculateGridSize(M0, M1);
                // (batch_size) * block_2_tile_map.CalculateGridSize(M0, M1);
            
            const bool in_out_same_vector_dim = GetMostContinousDim(arg.inStridesArray_[0]) == GetMostContinousDim(arg.outStridesArray_[0]);

            const auto kernel = in_out_same_vector_dim ? kernel_elementwise_1d<GridwiseElementwiseSameInOutVectorDim,
                                                                    InGridDescTuple,
                                                                    OutGridDescTuple,
                                                                    InDataTypePointerTuple,
                                                                    OutDataTypePointerTuple,
                                                                    Block2TileMap,
                                                                    ElementwiseOperation,
                                                                    UnaryOperation,
                                                                    Scale> :
                                                            kernel_elementwise_1d<GridwiseElementwise,
                                                                    InGridDescTuple,
                                                                    OutGridDescTuple,
                                                                    InDataTypePointerTuple,
                                                                    OutDataTypePointerTuple,
                                                                    Block2TileMap,
                                                                    ElementwiseOperation,
                                                                    UnaryOperation,
                                                                    Scale>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(grid_size),
                                                        dim3(BlockSize),
                                                        0,
                                                        in_grid_desc_tuple,
                                                        out_grid_desc_tuple,
                                                        arg.in_dev_buffers_,
                                                        arg.out_dev_buffers_,
                                                        block_2_tile_map,
                                                        arg.elementwise_op_,
                                                        arg.unary_op_,
                                                        arg.scale_op_,
                                                        batch_size);
            return elapsed_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
        // if(arg.lengths_.back() % MPerThread != 0)
        //     return false;

        auto IsScalarPerVectorValid = [&](const std::array<index_t, NumDim>& lengths,
                                          const std::array<index_t, NumDim>& strides,
                                          index_t scalarPerVector) {
            if (scalarPerVector == 1) {
                return true;
            }
            for (index_t d = 0; d < NumDim; d++) {
                if(strides[d] == 1 && lengths[d] % scalarPerVector == 0) {
                    return true;
                }
            }
            return false;
        };

        bool valid = true;
        static_for<0, NumInput, 1>{}([&](auto I) {
            if(!IsScalarPerVectorValid(
                   arg.lengths_, arg.inStridesArray_[I.value], InScalarPerVectorSeq::At(I)))
                valid = false;
        });

        static_for<0, NumOutput, 1>{}([&](auto I) {
            if(!IsScalarPerVectorValid(
                   arg.lengths_, arg.outStridesArray_[I.value], OutScalarPerVectorSeq::At(I)))
                valid = false;
        });

        return valid;
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const std::array<index_t, NumDim> lengths,
                 const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                 const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                 const std::array<const void*, NumInput> in_dev_buffers,
                 const std::array<void*, NumOutput> out_dev_buffers,
                 ElementwiseOperation elementwise_op,
                 UnaryOperation unary_op,
                 Scale scale_op)
    {
        return Argument{lengths,
                        inStridesArray,
                        outStridesArray,
                        in_dev_buffers,
                        out_dev_buffers,
                        elementwise_op,
                        unary_op,
                        scale_op};
    }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const std::array<index_t, NumDim> lengths,
                        const std::array<std::array<index_t, NumDim>, NumInput> inStridesArray,
                        const std::array<std::array<index_t, NumDim>, NumOutput> outStridesArray,
                        const std::array<const void*, NumInput> in_dev_buffers,
                        const std::array<void*, NumOutput> out_dev_buffers,
                        ElementwiseOperation elementwise_op,
                        UnaryOperation unary_op,
                        Scale scale_op) override
    {
        return std::make_unique<Argument>(lengths,
                                          inStridesArray,
                                          outStridesArray,
                                          in_dev_buffers,
                                          out_dev_buffers,
                                          elementwise_op,
                                          unary_op,
                                          scale_op);
    }

    static auto MakeInvoker() { return Invoker{}; }
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    };

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceElementwiseNormalizationImpl<";
        str << NumDim << ", ";
        str << M0PerThread << ", ";
        str << M1PerThread << ">";
        // clang-format on

        return str.str();
    }
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
