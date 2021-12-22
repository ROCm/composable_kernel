
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

template <typename TSrc, typename TComp, typename TDst>
class ReductionHost
{
    public:
    ReductionHost() = default;
    ReductionHost(const ck::ReduceTensorOp_t reduceOp_,
                  ck::NanPropagation_t nanOpt_,
                  ck::ReduceTensorIndices_t indiceOpt_,
                  HostTensorDescriptor& inDesc,
                  HostTensorDescriptor& outDesc,
                  const std::vector<int>& invariantDims_,
                  const std::vector<int>& toReduceDims_)
    {
        this->reduceOp    = reduceOp_;
        this->nanOpt      = nanOpt_;
        this->indiceOpt   = indiceOpt_;

        this->inLengths  = inDesc.GetLengths();
        this->outLengths = outDesc.GetLengths();
        this->inStrides  = inDesc.GetStrides();
        this->outStrides = outDesc.GetStrides();

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

    void Run(float alpha, const TSrc* in_data, float beta, TDst* out_data, int* indices)
    {
        if constexpr(std::is_same<TComp, float>::value)
        {
            if constexpr(std::is_same<TDst, double>::value)
                RunImpl<double>(alpha, in_data, beta, out_data, indices);
            else
                RunImpl<float>(alpha, in_data, beta, out_data, indices);
        }
        else if constexpr(std::is_same<TComp, float16>::value)
        {
            if constexpr(std::is_same<TDst, double>::value || std::is_same<TDst, float>::value)
                RunImpl<TDst>(alpha, in_data, beta, out_data, indices);
            else
                RunImpl<float16>(alpha, in_data, beta, out_data, indices);
        }
        else if constexpr (std::is_same<TComp, double>::value)
            RunImpl<double>(alpha, in_data, beta, out_data, indices);
        return;
    };

    private:
    ck::ReduceTensorOp_t reduceOp;

    ck::NanPropagation_t nanOpt;
    ck::ReduceTensorIndices_t indiceOpt;

    std::vector<size_t> inLengths;
    std::vector<size_t> outLengths;
    std::vector<size_t> inStrides;
    std::vector<size_t> outStrides;

    std::vector<size_t> invariantLengths;
    std::vector<size_t> toReduceLengths;

    std::vector<int> invariantDims;
    std::vector<int> toReduceDims;

    bool reduceAllDims;

    template <typename compType>
    void RunImpl(float alpha, const TSrc* in_data, float beta, TDst* out_data, int* indices)
    {
        bool need_indices = (indiceOpt == ck::ReduceTensorIndices_t::FLATTENED_INDICES) &&
                            (reduceOp == ck::ReduceTensorOp_t::MIN || reduceOp == ck::ReduceTensorOp_t::MAX ||
                             reduceOp == ck::ReduceTensorOp_t::AMAX);

        if(need_indices)
            RunImpl_with_indices<compType>(alpha, in_data, beta, out_data, indices);
        else
            RunImpl_no_indices<compType>(alpha, in_data, beta, out_data);
    };

    template <typename compType>
    void
    RunImpl_with_indices(float alpha, const TSrc* in_data, float beta, TDst* out_data, int* indices)
    {
        using reduce::PosUnaryOpFn;
        using reduce::PreUnaryOpFn;
        using reduce::ReduceOpFn2;
        using reduce::ReduceOpZeroVal;
        using reduce::binop_with_nan_check;
        using reduce::binop_with_nan_check2;
        using reduce::float_equal_one;
        using reduce::float_equal_zero;

        auto opReduce = ReduceOpFn2<compType>(this->reduceOp);

        int divider = 1;
        for(int i = 0; i < toReduceLengths.size(); i++)
            divider *= toReduceLengths[i];

        auto PreUnaryOp = PreUnaryOpFn<compType>(reduceOp, divider);
        auto PosUnaryOp = PosUnaryOpFn<compType>(reduceOp, divider);

        if(reduceAllDims)
        {
            std::vector<std::vector<size_t>> indexes_1;

            get_all_indexes(inLengths, 0, indexes_1); // generate the input indexes space

            auto accuVal  = ReduceOpZeroVal<compType>(this->reduceOp);
            int accuIndex = 0;

            // go through indexes of the invariant dimensions
            for(const auto& src_index : indexes_1)
            {
                auto src_offset = get_offset_from_index(this->inStrides, src_index);

                auto currVal = static_cast<compType>(in_data[src_offset]);

                // unary operation before reducing, needed by AMAX. For MIN/MAX, nothing is actually
                // done
                PreUnaryOp(currVal);

                auto currIndex = get_flatten_offset(inLengths, src_index);
                binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
            };

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= static_cast<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_zero(beta))
                accuVal += static_cast<compType>(out_data[0]) * static_cast<compType>(beta);

            // store the reduced value to dst location
            out_data[0] = static_cast<TDst>(accuVal);
            indices[0]  = accuIndex;
        }
        else
        {
            std::vector<std::vector<size_t>> indexes_1, indexes_2;

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

                compType accuVal = ReduceOpZeroVal<compType>(this->reduceOp);
                int accuIndex    = 0;

                // go through indexes of the toReduce dimensions
                for(const auto& index_2 : indexes_2)
                {
                    // generate the part of src index belonging to toReduce dims
                    for(int k = 0; k < toReduceDims.size(); k++)
                        src_index[toReduceDims[k]] = index_2[k];

                    auto src_offset = get_offset_from_index(this->inStrides, src_index);

                    auto currVal = static_cast<compType>(in_data[src_offset]);
                    // unary operation before reducing, needed by AMAX. For MIN/MAX, nothing is
                    // actually done
                    PreUnaryOp(currVal);

                    auto currIndex = get_flatten_offset(toReduceLengths, index_2);
                    binop_with_nan_check2(nanOpt, opReduce, accuVal, currVal, accuIndex, currIndex);
                };

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= static_cast<compType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal +=
                        static_cast<compType>(out_data[dst_offset]) * static_cast<compType>(beta);

                // store the reduced value to dst location
                out_data[dst_offset] = static_cast<TDst>(accuVal);
                indices[dst_offset]  = accuIndex;
            };
        };
    }; // end of RunImpl_with_indices()

    template <typename compType>
    void RunImpl_no_indices(float alpha, const TSrc* in_data, float beta, TDst* out_data)
    {
        using reduce::PosUnaryOpFn;
        using reduce::PreUnaryOpFn;
        using reduce::ReduceOpFn;
        using reduce::ReduceOpZeroVal;
        using reduce::binop_with_nan_check;
        using reduce::binop_with_nan_check2;
        using reduce::float_equal_one;
        using reduce::float_equal_zero;

        auto opReduce = ReduceOpFn<compType>(this->reduceOp);

        int divider = 1;
        for(int i = 0; i < toReduceLengths.size(); i++)
            divider *= toReduceLengths[i];

        auto PreUnaryOp = PreUnaryOpFn<compType>(reduceOp, divider);
        auto PosUnaryOp = PosUnaryOpFn<compType>(reduceOp, divider);

        if(reduceAllDims)
        {
            std::vector<std::vector<size_t>> indexes_1;

            get_all_indexes(inLengths, 0, indexes_1); // generate the input indexes space

            auto accuVal = ReduceOpZeroVal<compType>(this->reduceOp);

            // go through indexes of the invariant dimensions
            for(const auto& src_index : indexes_1)
            {
                auto src_offset = get_offset_from_index(this->inStrides, src_index);

                auto currVal = static_cast<compType>(in_data[src_offset]);

                PreUnaryOp(currVal);

                binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
            };

            PosUnaryOp(accuVal);

            // scale the accumulated value
            if(!float_equal_one(alpha))
                accuVal *= static_cast<compType>(alpha);

            // scale the prior dst value and add it to the accumulated value
            if(!float_equal_zero(beta))
                accuVal += static_cast<compType>(out_data[0]) * static_cast<compType>(beta);

            // store the reduced value to dst location
            out_data[0] = static_cast<TDst>(accuVal);
        }
        else
        {
            std::vector<std::vector<size_t>> indexes_1, indexes_2;

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

                compType accuVal = ReduceOpZeroVal<compType>(this->reduceOp);

                // go through indexes of the toReduce dimensions
                for(const auto& index_2 : indexes_2)
                {
                    // generate the part of src index belonging to toReduce dims
                    for(int k = 0; k < toReduceDims.size(); k++)
                        src_index[toReduceDims[k]] = index_2[k];

                    auto src_offset = get_offset_from_index(this->inStrides, src_index);

                    auto currVal = static_cast<compType>(in_data[src_offset]);

                    PreUnaryOp(currVal);

                    binop_with_nan_check(nanOpt, opReduce, accuVal, currVal);
                };

                PosUnaryOp(accuVal);

                // scale the accumulated value
                if(!float_equal_one(alpha))
                    accuVal *= static_cast<compType>(alpha);

                // scale the prior dst value and add it to the accumulated value
                if(!float_equal_zero(beta))
                    accuVal +=
                        static_cast<compType>(out_data[dst_offset]) * static_cast<compType>(beta);

                // store the reduced value to dst location
                out_data[dst_offset] = static_cast<TDst>(accuVal);
            };
        };
    }; // end of RunImpl_no_indices()
};

#endif
