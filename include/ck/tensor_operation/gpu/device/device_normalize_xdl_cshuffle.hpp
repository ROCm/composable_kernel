#pragma once
#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Input: x, E[x], E[x^2], Gamma, Beta
// Output: {[(x - E[x]) / sqrt(E[x^2] - E[x]^2 + epsilon)] * gamma} + beta
template <typename XDataType,
          typename MeanDataType,
          typename MeanSquareDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename OutDataType,
          typename ComputeDataType,
          typename OutElementwiseFunctor,
          index_t NDim,
          index_t MPerThread,
          index_t XScalarPerVector,
          index_t MeanScalarPerVector,
          index_t MeanSquareScalarPerVector,
          index_t GammaScalarPerVector,
          index_t BetaScalarPerVector>
struct DeviceNormalize_Xdl_CShuffle : public BaseOperator
{
    static constexpr auto I0 = Number<0>{};

    template <typename Desc_M>
    static auto PadDescriptor_M_1d(Desc_M desc_m, index_t gridSize, index_t blockSize)
    {
        const auto m            = desc_m.GetLength(I0);
        const index_t loop_step = gridSize * blockSize * MPerThread;
        const auto pad          = math::integer_least_multiple(m, loop_step) - m;
        const auto desc_m_pad =
            transform_tensor_descriptor(desc_m,
                                        make_tuple(make_right_pad_transform(m, pad)),
                                        make_tuple(Sequence<0>{}),
                                        make_tuple(Sequence<0>{}));
        return desc_m_pad;
    }

    static auto MakeDescriptor_M(const std::vector<index_t>& lengths,
                                 const std::vector<index_t>& stride,
                                 index_t gridSize,
                                 index_t blockSize)
    {
        auto tupleOfShape  = generate_tuple([&](auto I) { return lengths[I]; }, Number<NDim>{});
        auto tupleOfStride = generate_tuple([&](auto I) { return stride[I]; }, Number<NDim>{});

        // nd desc - [s0, s1, s2, ...]
        const auto desc = make_naive_tensor_descriptor(tupleOfShape, tupleOfStride);

        // merge nd to 1d desc - [s0 * s1 * ...]
        if constexpr(NDim > 1)
        {
            const auto desc_m = transform_tensor_descriptor(
                desc,
                make_tuple(make_merge_transform(tupleOfShape)),
                make_tuple(generate_sequence_v2([&](auto I) { return I; }, Number<NDim>{})),
                make_tuple(Sequence<0>{}));

            return PadDescriptor_M_1d(desc_m, gridSize, blockSize);
        }
        else
            return PadDescriptor_M_1d(desc, gridSize, blockSize);
    }

    using GridDesc_M = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));

    struct Argument : public BaseArgument
    {
        Argument(const XDataType* p_x,
                 const MeanDataType* p_mean,
                 const MeanSquareDataType* p_mean_square,
                 const GammaDataType* p_gamma,
                 const BetaDataType* p_beta,
                 OutDataType* p_output,
                 const std::vector<index_t>& lengths,
                 const std::vector<index_t>& stride_x,
                 const std::vector<index_t>& stride_mean,
                 const std::vector<index_t>& stride_mean_square,
                 const std::vector<index_t>& stride_gamma,
                 const std::vector<index_t>& stride_beta,
                 const std::vector<index_t>& stride_output,
                 OutElementwiseFunctor functor)
            : p_x_(p_x),
              p_mean_(p_mean),
              p_mean_square_(p_mean_square),
              p_gamma_(p_gamma),
              p_beta_(p_beta),
              p_output_(p_output),
              lengths_(lengths),
              stride_x_(stride_x),
              stride_mean_(stride_mean),
              stride_mean_square_(stride_mean_square),
              stride_gamma_(stride_gamma),
              stride_beta_(stride_beta),
              functor_(functor),
              blockSize_(256),
              gridSize_(120) // FIXME - Calculate the grid size by number of CU in the future
        {
            x_grid_desc_m_    = MakeDescriptor_M(lengths, stride_x, gridSize_, blockSize_);
            mean_grid_desc_m_ = MakeDescriptor_M(lengths, stride_mean, gridSize_, blockSize_);
            mean_square_grid_desc_m_ =
                MakeDescriptor_M(lengths, stride_mean_square, gridSize_, blockSize_);
            gamma_grid_desc_m_  = MakeDescriptor_M(lengths, stride_gamma, gridSize_, blockSize_);
            beta_grid_desc_m_   = MakeDescriptor_M(lengths, stride_beta, gridSize_, blockSize_);
            output_grid_desc_m_ = MakeDescriptor_M(lengths, stride_output, gridSize_, blockSize_);
        }

        const XDataType* p_x_;
        const MeanDataType* p_mean_;
        const MeanSquareDataType* p_mean_square_;
        const GammaDataType* p_gamma_;
        const BetaDataType* p_beta_;
        OutDataType* p_output_;
        std::vector<index_t> lengths_;
        GridDesc_M x_grid_desc_m_;
        GridDesc_M mean_grid_desc_m_;
        GridDesc_M mean_square_grid_desc_m_;
        GridDesc_M gamma_grid_desc_m_;
        GridDesc_M beta_grid_desc_m_;
        GridDesc_M output_grid_desc_m_;
        std::vector<index_t> stride_x_;
        std::vector<index_t> stride_mean_;
        std::vector<index_t> stride_mean_square_;
        std::vector<index_t> stride_gamma_;
        std::vector<index_t> stride_beta_;
        OutElementwiseFunctor functor_;
        index_t blockSize_;
        index_t gridSize_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            // TODO
            (void)arg;
            (void)stream_config;
            return 0;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        const Argument* pArg = dynamic_cast<const Argument*>(p_arg);

        if(pArg == nullptr)
            return false;

        if(pArg->lengths_.size() != NDim)
            return false;

        if(pArg->lengths_.back() % MPerThread != 0)
            return false;

        auto IsScalarPerVectorValid = [](bool isLastDimensionCoalesced, int scalarPerVector) {
            bool ret = true;

            if(!isLastDimensionCoalesced)
                ret = scalarPerVector == 1;
            else
                ret = MPerThread % scalarPerVector == 0;

            return ret;
        };

        if(!IsScalarPerVectorValid(pArg->stride_x_.back() == 1, XScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->stride_mean_.back() == 1, MeanScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->stride_mean_square_.back() == 1,
                                   MeanSquareScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->stride_gamma_.back() == 1, GammaScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->stride_beta_.back() == 1, BetaScalarPerVector))
            return false;
    };
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
