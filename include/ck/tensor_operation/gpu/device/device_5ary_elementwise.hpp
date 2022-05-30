#pragma once
#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "common_header.hpp"
#include "gridwise_5ary_Elementwise_1d.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename DDataType,
          typename EDataType,
          typename FDataType,
          typename ComputeDataType,
          typename ElementwiseFunctor,
          index_t NDim,
          index_t MPerThread,
          index_t AScalarPerVector,
          index_t BScalarPerVector,
          index_t CScalarPerVector,
          index_t DScalarPerVector,
          index_t EScalarPerVector,
          index_t FScalarPerVector>
struct Device5AryElementwise : public BaseOperator
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

    using AGridDesc_M = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));
    using BGridDesc_M = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));
    using CGridDesc_M = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));
    using DGridDesc_M = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));
    using EGridDesc_M = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));
    using FGridDesc_M = decltype(MakeDescriptor_M({1, 1}, {1, 1}, 1, 1));

    using Gridwise5AryEltwise = Gridwise5AryElementwise_1D<ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           DDataType,
                                                           EDataType,
                                                           FDataType,
                                                           ComputeDataType,
                                                           AGridDesc_M,
                                                           BGridDesc_M,
                                                           CGridDesc_M,
                                                           DGridDesc_M,
                                                           EGridDesc_M,
                                                           FGridDesc_M,
                                                           ElementwiseFunctor,
                                                           MPerThread,
                                                           AScalarPerVector,
                                                           BScalarPerVector,
                                                           CScalarPerVector,
                                                           DScalarPerVector,
                                                           EScalarPerVector,
                                                           FScalarPerVector>;

    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a,
                 const BDataType* p_b,
                 const CDataType* p_c,
                 const DDataType* p_d,
                 const EDataType* p_e,
                 FDataType* p_f,
                 const std::vector<index_t>& lengths,
                 const std::vector<index_t>& a_strides,
                 const std::vector<index_t>& b_strides,
                 const std::vector<index_t>& c_strides,
                 const std::vector<index_t>& d_strides,
                 const std::vector<index_t>& e_strides,
                 const std::vector<index_t>& f_strides,
                 ElementwiseFunctor functor)
            : p_a_(p_a),
              p_b_(p_b),
              p_c_(p_c),
              p_d_(p_d),
              p_e_(p_e),
              p_f_(p_f),
              lengths_(lengths),
              a_strides_(a_strides),
              b_strides_(b_strides),
              c_strides_(c_strides),
              d_strides_(d_strides),
              e_strides_(e_strides),
              f_strides_(f_strides),
              functor_(functor),
              blockSize_(256),
              gridSize_(120) // FIXME - Calculate the grid size by number of CU in the future
        {
            a_grid_desc_m_ = MakeDescriptor_M(lengths, a_strides, gridSize_, blockSize_);
            b_grid_desc_m_ = MakeDescriptor_M(lengths, b_strides, gridSize_, blockSize_);
            c_grid_desc_m_ = MakeDescriptor_M(lengths, c_strides, gridSize_, blockSize_);
            d_grid_desc_m_ = MakeDescriptor_M(lengths, d_strides, gridSize_, blockSize_);
            e_grid_desc_m_ = MakeDescriptor_M(lengths, e_strides, gridSize_, blockSize_);
            f_grid_desc_m_ = MakeDescriptor_M(lengths, f_strides, gridSize_, blockSize_);
        }

        const ADataType* p_a_;
        const BDataType* p_b_;
        const CDataType* p_c_;
        const DDataType* p_d_;
        const EDataType* p_e_;
        FDataType* p_f_;
        std::vector<index_t> lengths_;
        AGridDesc_M a_grid_desc_m_;
        BGridDesc_M b_grid_desc_m_;
        CGridDesc_M c_grid_desc_m_;
        DGridDesc_M d_grid_desc_m_;
        EGridDesc_M e_grid_desc_m_;
        FGridDesc_M f_grid_desc_m_;
        std::vector<index_t> a_strides_;
        std::vector<index_t> b_strides_;
        std::vector<index_t> c_strides_;
        std::vector<index_t> d_strides_;
        std::vector<index_t> e_strides_;
        std::vector<index_t> f_strides_;
        ElementwiseFunctor functor_;
        index_t blockSize_;
        index_t gridSize_;
    };

    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            const auto kernel = kernel_5ary_elementwise_1d<Gridwise5AryEltwise,
                                                           ADataType,
                                                           BDataType,
                                                           CDataType,
                                                           DDataType,
                                                           EDataType,
                                                           FDataType,
                                                           AGridDesc_M,
                                                           BGridDesc_M,
                                                           CGridDesc_M,
                                                           DGridDesc_M,
                                                           EGridDesc_M,
                                                           FGridDesc_M,
                                                           ElementwiseFunctor>;

            float elapsed_time = launch_and_time_kernel(stream_config,
                                                        kernel,
                                                        dim3(arg.gridSize_),
                                                        dim3(arg.blockSize_),
                                                        0,
                                                        arg.p_a_,
                                                        arg.p_b_,
                                                        arg.p_c_,
                                                        arg.p_d_,
                                                        arg.p_e_,
                                                        arg.p_f_,
                                                        arg.a_grid_desc_m_,
                                                        arg.b_grid_desc_m_,
                                                        arg.c_grid_desc_m_,
                                                        arg.d_grid_desc_m_,
                                                        arg.e_grid_desc_m_,
                                                        arg.f_grid_desc_m_,
                                                        arg.functor_);
            return elapsed_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    bool IsSupportedArgument(const BaseArgument& p_arg) { return IsSupportedArgument(&p_arg); }

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

        if(!IsScalarPerVectorValid(pArg->a_strides_.back() == 1, AScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->b_strides_.back() == 1, BScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->c_strides_.back() == 1, CScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->d_strides_.back() == 1, DScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->e_strides_.back() == 1, EScalarPerVector))
            return false;

        if(!IsScalarPerVectorValid(pArg->f_strides_.back() == 1, FScalarPerVector))
            return false;

        return true;
    };

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             const CDataType* p_c,
                             const DDataType* p_d,
                             const EDataType* p_e,
                             FDataType* p_f,
                             std::vector<index_t> lengths,
                             std::vector<index_t> a_strides,
                             std::vector<index_t> b_strides,
                             std::vector<index_t> c_strides,
                             std::vector<index_t> d_strides,
                             std::vector<index_t> e_strides,
                             std::vector<index_t> f_strides,
                             ElementwiseFunctor functor)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        p_d,
                        p_e,
                        p_f,
                        lengths,
                        a_strides,
                        b_strides,
                        c_strides,
                        d_strides,
                        e_strides,
                        f_strides,
                        functor};
    }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      const void* p_c,
                                                      const void* p_d,
                                                      const void* p_e,
                                                      void* p_f,
                                                      std::vector<index_t> lengths,
                                                      std::vector<index_t> a_strides,
                                                      std::vector<index_t> b_strides,
                                                      std::vector<index_t> c_strides,
                                                      std::vector<index_t> d_strides,
                                                      std::vector<index_t> e_strides,
                                                      std::vector<index_t> f_strides,
                                                      ElementwiseFunctor functor)
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<const CDataType*>(p_c),
                                          static_cast<const DDataType*>(p_d),
                                          static_cast<const EDataType*>(p_e),
                                          static_cast<FDataType*>(p_f),
                                          lengths,
                                          a_strides,
                                          b_strides,
                                          c_strides,
                                          d_strides,
                                          e_strides,
                                          f_strides,
                                          functor);
    }

    static auto MakeInvoker() { return Invoker{}; }
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() { return std::make_unique<Invoker>(); }
}; // namespace device

} // namespace device
} // namespace tensor_operation
} // namespace ck
