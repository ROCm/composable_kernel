
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef HOST_REDUCTION_HPP_
#define HOST_REDUCTION_HPP_

#include <vector>
#include <array>
#include <functional>

#include "reduction_enums.hpp"
#include "host_reduce_util.hpp"
#include "host_tensor.hpp"
#include "data_type.hpp"

template <int NDim>
static void get_all_indexes(const std::array<size_t, NDim>& dimLengths,
                            std::vector<std::array<size_t, NDim>>& indexes)
{
    static_assert(NDim >= 1, "NDim >= 1 is required to use this function!");

    if constexpr(NDim == 1)
    {
        for(size_t i = 0; i < dimLengths[0]; i++)
        {
            std::array<size_t, 1> index{i};

            indexes.push_back(index);
        };
    }
    else
    {
        std::array<size_t, NDim - 1> partial_dim_lengths;

        for(int i = 0; i < NDim - 1; i++)
            partial_dim_lengths[i] = dimLengths[i + 1];

        std::vector<std::array<size_t, NDim - 1>> partial_indexes;

        get_all_indexes<NDim - 1>(partial_dim_lengths, partial_indexes);

        for(size_t i = 0; i < dimLengths[0]; i++)
            for(const auto& index : partial_indexes)
            {
                std::array<size_t, NDim> extIndex;

                extIndex[0] = i;

                for(int k = 0; k < NDim - 1; k++)
                    extIndex[k + 1] = index[k];

                indexes.push_back(extIndex);
            };
    };
};

template <int NDim>
static size_t get_offset_from_index(const std::array<size_t, NDim>& strides,
                                    const std::array<size_t, NDim>& index)
{
    size_t offset = 0;

    for(int i = 0; i < NDim; i++)
        offset += strides[i] * index[i];

    return (offset);
};

template <int NDim>
static size_t get_offset_from_index(const std::vector<size_t>& strides,
                                    const std::array<size_t, NDim>& index)
{
    size_t offset = 0;

    for(int i = 0; i < NDim; i++)
        offset += strides[i] * index[i];

    return (offset);
};

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          ck::ReduceTensorOp ReduceOpId,
          int Rank,
          int NumReduceDim,
          bool PropagateNan,
          bool NeedIndices>
struct ReductionHost
{
    using IndexDataType = int32_t;

    static constexpr int NumInvariantDim = Rank - NumReduceDim;

    std::vector<size_t> outStrides;
    std::vector<int> invariantDims;
    std::vector<int> reduceDims;

    IndexDataType divider;
    std::function<void(AccDataType&)> preUnaryOp;
    std::function<void(AccDataType&)> posUnaryOp;
    std::array<size_t, NumReduceDim> reduceLengths;
    std::array<size_t, NumReduceDim> reduceStrides;
    std::array<size_t, NumInvariantDim> invariantLengths;
    std::array<size_t, NumInvariantDim> invariantStrides;

    std::vector<std::array<size_t, NumReduceDim>> reduce_dim_indexes;
    std::vector<std::array<size_t, NumInvariantDim>> invariant_dim_indexes;

    ReductionHost(HostTensorDescriptor& inDesc,
                  HostTensorDescriptor& outDesc,
                  const std::vector<int>& invariantDims_,
                  const std::vector<int>& reduceDims_)
    {
        using ck::host_reduce::PosUnaryOpFn;
        using ck::host_reduce::PreUnaryOpFn;

        // this->outLengths = to_int_vector(outDesc.GetLengths());
        this->outStrides = outDesc.GetStrides();

        this->invariantDims = invariantDims_;
        this->reduceDims    = reduceDims_;

        int product = 1;

        for(int i = 0; i < NumReduceDim; i++)
        {
            reduceLengths[i] = inDesc.GetLengths()[reduceDims[i]];
            reduceStrides[i] = inDesc.GetStrides()[reduceDims[i]];
            product *= inDesc.GetLengths()[reduceDims[i]];
        };

        divider = product;

        for(int i = 0; i < NumInvariantDim; i++)
        {
            invariantLengths[i] = inDesc.GetLengths()[invariantDims[i]];
            invariantStrides[i] = inDesc.GetStrides()[invariantDims[i]];
        };

        reduce_dim_indexes.clear();
        get_all_indexes<NumReduceDim>(reduceLengths, reduce_dim_indexes);

        if constexpr(NumInvariantDim > 0)
        {
            invariant_dim_indexes.clear();
            get_all_indexes<NumInvariantDim>(invariantLengths, invariant_dim_indexes);
        };

        preUnaryOp = PreUnaryOpFn<AccDataType, ReduceOpId>(divider);
        posUnaryOp = PosUnaryOpFn<AccDataType, ReduceOpId>(divider);
    };

    void Run(float alpha,
             const InDataType* in_data,
             float beta,
             OutDataType* out_data,
             IndexDataType* out_indices)
    {
        if constexpr(NeedIndices)
        {
            RunImpl_with_index(alpha, in_data, beta, out_data, out_indices);
        }
        else
        {
            RunImpl_no_index(alpha, in_data, beta, out_data);
        };
    };

    void RunImpl_with_index(float alpha,
                            const InDataType* in_data,
                            float beta,
                            OutDataType* out_data,
                            IndexDataType* out_indices)
    {
        using ck::type_convert;
        using ck::host_reduce::binop_with_nan_check2;
        using ck::host_reduce::float_equal_one;
        using ck::host_reduce::float_equal_zero;
        using ck::host_reduce::ReduceOpFn2;
        using ck::host_reduce::ReduceOpZeroVal;

        auto opReduce2 = ReduceOpFn2<AccDataType, ReduceOpId>();

        if constexpr(NumInvariantDim == 0)
        {
            AccDataType accuVal     = ReduceOpZeroVal<AccDataType, ReduceOpId>();
            IndexDataType accuIndex = 0;

            for(IndexDataType i = 0; i < reduce_dim_indexes.size(); i++)
            {
                auto offset_reduce =
                    get_offset_from_index<NumReduceDim>(reduceStrides, reduce_dim_indexes[i]);

                auto currVal = type_convert<AccDataType>(in_data[offset_reduce]);

                preUnaryOp(currVal);

                auto currIndex = i;

                binop_with_nan_check2<AccDataType, PropagateNan>(
                    opReduce2, accuVal, currVal, accuIndex, currIndex);
            };

            posUnaryOp(accuVal);

            if(!float_equal_one(alpha))
                accuVal *= type_convert<AccDataType>(alpha);

            if(!float_equal_zero(beta))
                accuVal += type_convert<AccDataType>(out_data[0]) * type_convert<AccDataType>(beta);

            out_data[0]    = type_convert<OutDataType>(accuVal);
            out_indices[0] = accuIndex;
        }
        else
        {
            auto thread_reduce_func = [&](auto invariant_index) {
                AccDataType accuVal     = ReduceOpZeroVal<AccDataType, ReduceOpId>();
                IndexDataType accuIndex = 0;

                auto offset_invariant =
                    get_offset_from_index<NumInvariantDim>(invariantStrides, invariant_index);

                for(IndexDataType i = 0; i < reduce_dim_indexes.size(); i++)
                {
                    auto offset_reduce =
                        get_offset_from_index<NumReduceDim>(reduceStrides, reduce_dim_indexes[i]);

                    auto currVal =
                        type_convert<AccDataType>(in_data[offset_invariant + offset_reduce]);

                    preUnaryOp(currVal);

                    auto currIndex = i;

                    binop_with_nan_check2<AccDataType, PropagateNan>(
                        opReduce2, accuVal, currVal, accuIndex, currIndex);
                };

                posUnaryOp(accuVal);

                if(!float_equal_one(alpha))
                    accuVal *= type_convert<AccDataType>(alpha);

                auto dst_offset =
                    get_offset_from_index<NumInvariantDim>(outStrides, invariant_index);

                if(!float_equal_zero(beta))
                    accuVal += type_convert<AccDataType>(out_data[dst_offset]) *
                               type_convert<AccDataType>(beta);

                out_data[dst_offset]    = type_convert<OutDataType>(accuVal);
                out_indices[dst_offset] = accuIndex;
            };

            std::size_t num_thread = 1;
            std::size_t work_per_thread =
                (invariant_dim_indexes.size() + num_thread - 1) / num_thread;

            std::vector<joinable_thread> threads(num_thread);

            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t iw_begin = it * work_per_thread;
                std::size_t iw_end =
                    std::min((it + 1) * work_per_thread, invariant_dim_indexes.size());

                auto f = [=] {
                    for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                    {
                        thread_reduce_func(invariant_dim_indexes[iw]);
                    }
                };

                threads[it] = joinable_thread(f);
            }
        };
    };

    void RunImpl_no_index(float alpha, const InDataType* in_data, float beta, OutDataType* out_data)
    {
        using ck::type_convert;
        using ck::host_reduce::binop_with_nan_check;
        using ck::host_reduce::float_equal_one;
        using ck::host_reduce::float_equal_zero;
        using ck::host_reduce::ReduceOpFn;
        using ck::host_reduce::ReduceOpZeroVal;

        auto opReduce = ReduceOpFn<AccDataType, ReduceOpId>();

        if constexpr(NumInvariantDim == 0)
        {
            AccDataType accuVal = ReduceOpZeroVal<AccDataType, ReduceOpId>();

            for(const auto& reduce_index : reduce_dim_indexes)
            {
                auto offset_reduce =
                    get_offset_from_index<NumReduceDim>(reduceStrides, reduce_index);

                auto currVal = type_convert<AccDataType>(in_data[offset_reduce]);

                preUnaryOp(currVal);

                binop_with_nan_check<AccDataType, PropagateNan>(opReduce, accuVal, currVal);
            };

            posUnaryOp(accuVal);

            if(!float_equal_one(alpha))
                accuVal *= type_convert<AccDataType>(alpha);

            if(!float_equal_zero(beta))
                accuVal += type_convert<AccDataType>(out_data[0]) * type_convert<AccDataType>(beta);

            out_data[0] = type_convert<OutDataType>(accuVal);
        }
        else
        {
            auto thread_reduce_func = [&](auto invariant_index) {
                AccDataType accuVal = ReduceOpZeroVal<AccDataType, ReduceOpId>();

                auto offset_invariant =
                    get_offset_from_index<NumInvariantDim>(invariantStrides, invariant_index);

                for(const auto& reduce_index : reduce_dim_indexes)
                {
                    auto offset_reduce =
                        get_offset_from_index<NumReduceDim>(reduceStrides, reduce_index);

                    auto currVal =
                        type_convert<AccDataType>(in_data[offset_invariant + offset_reduce]);

                    preUnaryOp(currVal);

                    binop_with_nan_check<AccDataType, PropagateNan>(opReduce, accuVal, currVal);
                };

                posUnaryOp(accuVal);

                if(!float_equal_one(alpha))
                    accuVal *= type_convert<AccDataType>(alpha);

                auto dst_offset =
                    get_offset_from_index<NumInvariantDim>(outStrides, invariant_index);

                if(!float_equal_zero(beta))
                    accuVal += type_convert<AccDataType>(out_data[dst_offset]) *
                               type_convert<AccDataType>(beta);

                out_data[dst_offset] = type_convert<OutDataType>(accuVal);
            };

            std::size_t num_thread = 1;
            std::size_t work_per_thread =
                (invariant_dim_indexes.size() + num_thread - 1) / num_thread;

            std::vector<joinable_thread> threads(num_thread);

            for(std::size_t it = 0; it < num_thread; ++it)
            {
                std::size_t iw_begin = it * work_per_thread;
                std::size_t iw_end =
                    std::min((it + 1) * work_per_thread, invariant_dim_indexes.size());

                auto f = [=] {
                    for(std::size_t iw = iw_begin; iw < iw_end; ++iw)
                    {
                        thread_reduce_func(invariant_dim_indexes[iw]);
                    }
                };

                threads[it] = joinable_thread(f);
            }
        };
    };
};

#endif
