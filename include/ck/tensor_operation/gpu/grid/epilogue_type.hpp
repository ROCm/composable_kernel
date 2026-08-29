// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_reduce_wmma.hpp"
#include "ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_welford_wmma.hpp"
#include "ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma.hpp"
#include "ck/tensor_operation/gpu/grid/epilogue_direct_store.hpp"

namespace ck {

enum class EpilogueType
{
    CShuffle = 0,
    DirectStore,
    WelfordCShuffle,
    ReduceCShuffle
};

template <EpilogueType type, typename GridwiseGemm, typename ReduceTrait = Tuple<>>
struct get_epilogue
{
    private:
    static constexpr auto get_epilogue_implementation()
    {
        static_assert((type == EpilogueType::ReduceCShuffle) ==
                          (!std::is_same_v<ReduceTrait, Tuple<>>),
                      "Provide a ReduceTrait only if the desired epilogue type is ReduceCShuffle.");
        using TypeExtractor = typename GridwiseGemm::Traits;

        if constexpr(type == EpilogueType::CShuffle)
        {
            return EpilogueCShuffle<
                typename TypeExtractor::DsDataType_,
                typename TypeExtractor::EDataType_,
                typename TypeExtractor::AccDataType_,
                typename TypeExtractor::CShuffleDataType_,
                TypeExtractor::MPerBlock_,
                TypeExtractor::NPerBlock_,
                TypeExtractor::MPerWmma_,
                TypeExtractor::NPerWmma_,
                TypeExtractor::MRepeat_,
                TypeExtractor::NRepeat_,
                TypeExtractor::CShuffleMRepeatPerShuffle_,
                TypeExtractor::CShuffleNRepeatPerShuffle_,
                typename TypeExtractor::
                    CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
                typename TypeExtractor::CDEShuffleBlockTransferScalarPerVectors_,
                typename TypeExtractor::CDEElementwiseOperation_,
                typename TypeExtractor::ThisThreadBlock_,
                typename TypeExtractor::BlockwiseGemmPipe_>{};
        }
        else if constexpr(type == EpilogueType::DirectStore)
        {
            return EpilogueDirectStore<typename TypeExtractor::DsDataType_,
                                       typename TypeExtractor::EDataType_,
                                       typename TypeExtractor::AccDataType_,
                                       TypeExtractor::MRepeat_,
                                       TypeExtractor::NRepeat_,
                                       typename TypeExtractor::CDEElementwiseOperation_,
                                       typename TypeExtractor::BlockwiseGemmPipe_>{};
        }
        else if constexpr(type == EpilogueType::WelfordCShuffle)
        {
            return EpilogueWelfordCShuffle<
                typename TypeExtractor::DsDataType_,
                typename TypeExtractor::EDataType_,
                typename TypeExtractor::AccDataType_,
                typename TypeExtractor::CShuffleDataType_,
                TypeExtractor::MPerBlock_,
                TypeExtractor::NPerBlock_,
                TypeExtractor::MPerWmma_,
                TypeExtractor::NPerWmma_,
                TypeExtractor::MRepeat_,
                TypeExtractor::NRepeat_,
                TypeExtractor::CShuffleMRepeatPerShuffle_,
                TypeExtractor::CShuffleNRepeatPerShuffle_,
                typename TypeExtractor::
                    CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
                typename TypeExtractor::CDEShuffleBlockTransferScalarPerVectors_,
                typename TypeExtractor::CDEElementwiseOperation_,
                typename TypeExtractor::ThisThreadBlock_,
                typename TypeExtractor::BlockwiseGemmPipe_,
                TypeExtractor::BlockSize_>{{}, {}, {}, {}, {}};
        }
        else if constexpr(type == EpilogueType::ReduceCShuffle)
        {
            return EpilogueReduceCShuffle<
                typename TypeExtractor::DsDataType_,
                typename TypeExtractor::EDataType_,
                typename TypeExtractor::AccDataType_,
                typename TypeExtractor::CShuffleDataType_,
                TypeExtractor::MPerBlock_,
                TypeExtractor::NPerBlock_,
                TypeExtractor::MPerWmma_,
                TypeExtractor::NPerWmma_,
                TypeExtractor::MRepeat_,
                TypeExtractor::NRepeat_,
                TypeExtractor::CShuffleMRepeatPerShuffle_,
                TypeExtractor::CShuffleNRepeatPerShuffle_,
                typename TypeExtractor::
                    CDEShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock_,
                typename TypeExtractor::CDEShuffleBlockTransferScalarPerVectors_,
                typename TypeExtractor::CDEElementwiseOperation_,
                typename TypeExtractor::ThisThreadBlock_,
                typename TypeExtractor::BlockwiseGemmPipe_,
                TypeExtractor::GemmSpec_,
                TypeExtractor::BlockSize_,
                ReduceTrait>{{}, {}, {}, {}, {}};
        }
        else
        {
            static_assert(false, "Not implemented for the specified type.");
        }
    }

    public:
    using Type = decltype(get_epilogue_implementation());
};

template <EpilogueType type, typename GridwiseGemm, typename ReduceTrait = Tuple<>>
using get_epilogue_t = typename get_epilogue<type, GridwiseGemm, ReduceTrait>::Type;

} // namespace ck
