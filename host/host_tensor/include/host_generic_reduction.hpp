
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
#ifndef HOST_GENERIC_REDUCTION_HPP_
#define HOST_GENERIC_REDUCTION_HPP_

#include <vector>
#include <functional>
#include <limits>
#include <type_traits>
#include <cassert>
#include <cmath>

#include "reduction_enums.hpp"
#include "host_reduce_util.hpp"

using float16 = half_float::half;

namespace ck {

namespace host_reduce {

template <typename T>
static void
get_all_indexes(const std::vector<T>& dimLengths, int dim, std::vector<std::vector<T>>& indexes)
{
    if(dim < dimLengths.size())
    {
        std::vector<std::vector<T>> updated_indexes;

        if(dim == 0)
        {
            assert(indexes.size() == 0);
            assert(dimLengths[dim] > 0);
            for(T i = 0; i < dimLengths[dim]; i++)
            {
                std::vector<T> index = {i};

                updated_indexes.push_back(index);
            };
        }
        else
        {
            // go through all the current indexes
            for(const auto& index : indexes)
                for(T i = 0; i < dimLengths[dim]; i++)
                {
                    auto index_new = index;
                    index_new.push_back(i);

                    updated_indexes.push_back(index_new);
                };
        };

        // update to the indexes (output)
        indexes = updated_indexes;

        // further to construct the indexes from the updated status
        get_all_indexes(dimLengths, dim + 1, indexes);
    };
};

template <typename T>
static T get_offset_from_index(const std::vector<T>& strides, const std::vector<T>& index)
{
    T offset = 0;

    assert(strides.size() == index.size());

    for(int i = 0; i < index.size(); i++)
        offset += strides[i] * static_cast<T>(index[i]);

    return (offset);
};

template <typename T>
static T get_flatten_offset(const std::vector<T>& lengths, const std::vector<T>& index)
{
    T offset = 0;

    assert(lengths.size() == index.size() && lengths.size() > 0);

    int len  = lengths.size();
    T stride = 1;

    // for len==1, the loop is not executed
    for(int i = len - 1; i > 0; i--)
    {
        offset += stride * static_cast<T>(index[i]);

        stride *= lengths[i];
    };

    offset += stride * static_cast<T>(index[0]);

    return (offset);
};

template <typename InDataType,
          typename AccDataType,
          typename OutDataType,
          ck::ReduceTensorOp_t ReduceOpId,
          bool PropagateNan,
          bool NeedIndices>
class ReductionHost
{
    public:
    ReductionHost() = default;
    ReductionHost(HostTensorDescriptor& inDesc,
                  HostTensorDescriptor& outDesc,
                  const std::vector<int>& invariantDims_,
                  const std::vector<int>& toReduceDims_)
    {
        this->inLengths  = to_int_vector(inDesc.GetLengths());
        this->outLengths = to_int_vector(outDesc.GetLengths());
        this->inStrides  = to_int_vector(inDesc.GetStrides());
        this->outStrides = to_int_vector(outDesc.GetStrides());

        this->invariantDims = invariantDims_;
        this->toReduceDims  = toReduceDims_;

        assert(this->inLengths.size() == this->outLengths.size());
        assert(!this->toReduceDims.empty());

        for(const auto dim : this->invariantDims)
            this->invariantLengths.push_back(this->inLengths[dim]);

        for(const auto dim : this->toReduceDims)
            toReduceLengths.push_back(this->inLengths[dim]);

        this->reduceAllDims = this->invariantDims.empty();
    };

    ~ReductionHost(){};

    void
    Run(float alpha, const InDataType* in_data, float beta, OutDataType* out_data, int* indices)
    {
        if constexpr(NeedIndices)
            RunImpl_with_indices(alpha, in_data, beta, out_data, indices);
        else
            RunImpl_no_indices(alpha, in_data, beta, out_data);
    };

    private:
    std::vector<int> inLengths;
    std::vector<int> outLengths;
    std::vector<int> inStrides;
    std::vector<int> outStrides;

    std::vector<int> invariantLengths;
    std::vector<int> toReduceLengths;

    std::vector<int> invariantDims;
    std::vector<int> toReduceDims;

    bool reduceAllDims;

    void RunImpl_with_indices(
        float alpha, const InDataType* in_data, float beta, OutDataType* out_data, int* indices)
    {
        using ck::host_reduce::binop_with_nan_check;
        using ck::host_reduce::binop_with_nan_check2;
        using ck::host_reduce::float_equal_one;
        using ck::host_reduce::float_equal_zero;
        using ck::host_reduce::PosUnaryOpFn;
        using ck::host_reduce::PreUnaryOpFn;
        using ck::host_reduce::ReduceOpFn2;
        using ck::host_reduce::ReduceOpZeroVal;

        auto opReduce = ReduceOpFn2<AccDataType, ReduceOpId>();

        int divider = 1;
        for(int i = 0; i < toReduceLengths.size(); i++)
            divider *= toReduceLengths[i];

        auto PreUnaryOp = PreUnaryOpFn<AccDataType, ReduceOpId>(divider);
        auto PosUnaryOp = PosUnaryOpFn<AccDataType, ReduceOpId>(divider);

        if(reduceAllDims)
        {
            std::vector<std::vector<int>> indexes_1;

            get_all_indexes(inLengths, 0, indexes_1); // generate the input indexes space

            auto accuVal  = ReduceOpZeroVal<AccDataType, ReduceOpId>();
            int accuIndex = 0;

            // go through indexes of the invariant dimensions
            for(const auto& src_index : indexes_1)
            {
                auto src_offset = get_offset_from_index(this->inStrides, src_index);

                auto currVal = static_cast<AccDataType>(in_data[src_offset]);

                // unary operation before reducing, needed by AMAX. For MIN/MAX, nothing is actually
                // done
                PreUnaryOp(currVal);

                auto currIndex = get_flatten_offset(inLengths, src_index);
                binop_with_nan_check2<AccDataType, PropagateNan>(
                    opReduce, accuVal, currVal, accuIndex, currIndex);
            };

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= static_cast<AccDataType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_zero(beta))
                accuVal += static_cast<AccDataType>(out_data[0]) * static_cast<AccDataType>(beta);

            // store the reduced value to dst location
            out_data[0] = static_cast<OutDataType>(accuVal);
            indices[0]  = accuIndex;
        }
        else
        {
            std::vector<std::vector<int>> indexes_1, indexes_2;

            get_all_indexes(
                this->invariantLengths, 0, indexes_1); // generate the invariant indexes space
            get_all_indexes(
                this->toReduceLengths, 0, indexes_2); // generate the toReduce indexes space

            // go through indexes of the invariant dimensions
            for(const auto& index_1 : indexes_1)
            {
                std::vector<int> src_index;
                std::vector<int> dst_index;

                src_index.resize(this->inLengths.size());

                // generate the part of src index belonging to invariant dims
                for(int k = 0; k < invariantDims.size(); k++)
                    src_index[invariantDims[k]] = index_1[k];

                for(int k = 0; k < invariantDims.size(); k++)
                    dst_index.push_back(index_1[k]);

                int dst_offset = get_offset_from_index(this->outStrides, dst_index);

                AccDataType accuVal = ReduceOpZeroVal<AccDataType, ReduceOpId>();
                int accuIndex       = 0;

                // go through indexes of the toReduce dimensions
                for(const auto& index_2 : indexes_2)
                {
                    // generate the part of src index belonging to toReduce dims
                    for(int k = 0; k < toReduceDims.size(); k++)
                        src_index[toReduceDims[k]] = index_2[k];

                    auto src_offset = get_offset_from_index(this->inStrides, src_index);

                    auto currVal = static_cast<AccDataType>(in_data[src_offset]);
                    // unary operation before reducing, needed by AMAX. For MIN/MAX, nothing is
                    // actually done
                    PreUnaryOp(currVal);

                    auto currIndex = get_flatten_offset(toReduceLengths, index_2);
                    binop_with_nan_check2<AccDataType, PropagateNan>(
                        opReduce, accuVal, currVal, accuIndex, currIndex);
                };

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= static_cast<AccDataType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal += static_cast<AccDataType>(out_data[dst_offset]) *
                               static_cast<AccDataType>(beta);

                // store the reduced value to dst location
                out_data[dst_offset] = static_cast<OutDataType>(accuVal);
                indices[dst_offset]  = accuIndex;
            };
        };
    }; // end of RunImpl_with_indices()

    void
    RunImpl_no_indices(float alpha, const InDataType* in_data, float beta, OutDataType* out_data)
    {
        using ck::host_reduce::binop_with_nan_check;
        using ck::host_reduce::binop_with_nan_check2;
        using ck::host_reduce::float_equal_one;
        using ck::host_reduce::float_equal_zero;
        using ck::host_reduce::PosUnaryOpFn;
        using ck::host_reduce::PreUnaryOpFn;
        using ck::host_reduce::ReduceOpFn;
        using ck::host_reduce::ReduceOpZeroVal;

        auto opReduce = ReduceOpFn<AccDataType, ReduceOpId>();

        int divider = 1;
        for(int i = 0; i < toReduceLengths.size(); i++)
            divider *= toReduceLengths[i];

        auto PreUnaryOp = PreUnaryOpFn<AccDataType, ReduceOpId>(divider);
        auto PosUnaryOp = PosUnaryOpFn<AccDataType, ReduceOpId>(divider);

        if(reduceAllDims)
        {
            std::vector<std::vector<int>> indexes_1;

            get_all_indexes(inLengths, 0, indexes_1); // generate the input indexes space

            auto accuVal = ReduceOpZeroVal<AccDataType, ReduceOpId>();

            // go through indexes of the invariant dimensions
            for(const auto& src_index : indexes_1)
            {
                auto src_offset = get_offset_from_index(this->inStrides, src_index);

                auto currVal = static_cast<AccDataType>(in_data[src_offset]);

                PreUnaryOp(currVal);

                binop_with_nan_check<AccDataType, PropagateNan>(opReduce, accuVal, currVal);
            };

            PosUnaryOp(accuVal);

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= static_cast<AccDataType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_zero(beta))
                accuVal += static_cast<AccDataType>(out_data[0]) * static_cast<AccDataType>(beta);

            // store the reduced value to dst location
            out_data[0] = static_cast<OutDataType>(accuVal);
        }
        else
        {
            std::vector<std::vector<int>> indexes_1, indexes_2;

            get_all_indexes(
                this->invariantLengths, 0, indexes_1); // generate the invariant indexes space
            get_all_indexes(
                this->toReduceLengths, 0, indexes_2); // generate the toReduce indexes space

            // go through indexes of the invariant dimensions
            for(const auto& index_1 : indexes_1)
            {
                std::vector<int> src_index;
                std::vector<int> dst_index;

                src_index.resize(this->inLengths.size());

                for(int k = 0; k < invariantDims.size(); k++)
                    dst_index.push_back(index_1[k]);

                int dst_offset = get_offset_from_index(this->outStrides, dst_index);

                // generate the part of src index belonging to invariant dims
                for(int k = 0; k < invariantDims.size(); k++)
                    src_index[invariantDims[k]] = index_1[k];

                AccDataType accuVal = ReduceOpZeroVal<AccDataType, ReduceOpId>();

                // go through indexes of the toReduce dimensions
                for(const auto& index_2 : indexes_2)
                {
                    // generate the part of src index belonging to toReduce dims
                    for(int k = 0; k < toReduceDims.size(); k++)
                        src_index[toReduceDims[k]] = index_2[k];

                    auto src_offset = get_offset_from_index(this->inStrides, src_index);

                    auto currVal = static_cast<AccDataType>(in_data[src_offset]);

                    PreUnaryOp(currVal);

                    binop_with_nan_check<AccDataType, PropagateNan>(opReduce, accuVal, currVal);
                };

                PosUnaryOp(accuVal);

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= static_cast<AccDataType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal += static_cast<AccDataType>(out_data[dst_offset]) *
                               static_cast<AccDataType>(beta);

                // store the reduced value to dst location
                out_data[dst_offset] = static_cast<OutDataType>(accuVal);
            };
        };
    }; // end of RunImpl_no_indices()
};

}; // end of namespace host_reduce

}; // end of namespace ck

#endif
