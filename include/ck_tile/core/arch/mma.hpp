// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "mfma/mfma.hpp"
#include "wmma/wmma.hpp"

#include "mma_traits.hpp"

namespace ck::tile::core::arch
{
    enum struct MmaAccumPolicy : uint32_t
    {
        ROW_MAJOR = 0u,
        COL_MAJOR = 1u,
    };

    /*! \class Mma
    *   \brief Driver for the wave-tile Mma operation. Given a backend block-wise mma implementation (e.g., mfma or wmma),
    * this class performs block-wise decomposition to matrix-multiply input fragments of (A: FragM x FragK) x (B: FragK x FragN)
    * and accumulates results into output fragment (C: FragM x FragN).
    *  @tparam FragM Mma fragment M dimension
    *  @tparam FragN Mma fragment K dimension
    *  @tparam FragK Mma fragment M dimension
    *  @tparam MmaImpl The backend wrapper class that will perform block-wise mma op (e.g., mfma or wmma)
    *  @tparam PreOps The pre-mma transform operations to be applied to input fragments A, B and C
    *  @tparam PostOps The post-mma transform operations to be applied to the output fragment D
    *  @tparam MmaAccumPolicy The block order of the accumulation registers (row major or col major block order)
    */ 
    // NOTE: The selector classes here are an automated suggestion to promote ease of use.
    // Users can always directly specify the MmaImpl, PreOps, PostOps template parameters
    // if they wish to bypass the selection logic.
    template <typename DataTypeA,
              typename DataTypeB,
              typename ComputeT,
              uint32_t FragM,
              uint32_t FragN,
              uint32_t FragK,
              class MmaImpl = typename MmaSelector<DataTypeA, DataTypeB, ComputeT, FragM, FragN, FragK>::SelectedOp,
              class PreOps = typename TransformSelector<MmaImpl>::Pre,
              class PostOps = typename TransformSelector<MmaImpl>::Post,
              MmaAccumPolicy AccumPolicy = MmaAccumPolicy::ROW_MAJOR>
    struct Mma
    {
        using BlockWiseMma = MmaImpl;
        using BlockWiseMmaTraits = MmaTraits<BlockWiseMma>;

        // Block dimensions
        constexpr static uint32_t BlockM = BlockWiseMmaTraits::BlockM;
        constexpr static uint32_t BlockN = BlockWiseMmaTraits::BlockN;
        constexpr static uint32_t BlockK = BlockWiseMmaTraits::BlockK;

        // Block vector dimensions (packed registers as input to impl)
        constexpr static uint32_t BlockSizeA = BlockWiseMmaTraits::BlockSizeA;
        constexpr static uint32_t BlockSizeB = BlockWiseMmaTraits::BlockSizeB;
        constexpr static uint32_t BlockSizeC = BlockWiseMmaTraits::BlockSizeC;

        // Block counts for decomposition
        constexpr static uint32_t BlocksM = FragM / BlockM;
        constexpr static uint32_t BlocksN = FragN / BlockN;
        constexpr static uint32_t BlocksK = FragK / BlockK;
        constexpr static uint32_t BlocksC = BlocksM * BlocksN;

        // Register grouping size for accum
        constexpr static uint32_t AccumRowSize = BlocksN * BlockSizeC;
        constexpr static uint32_t AccumColSize = BlocksM * BlockSizeC;

        // Sanity checks
        static_assert(FragM >= BlockM, "FragM must be larger than BlockM");
        static_assert(FragN >= BlockN, "FragN must be larger than BlockN");
        static_assert(FragK >= BlockK, "FragK must be larger than BlockK");
        static_assert(FragM % BlockM == 0u, "FragM must be a multiple of BlockM");
        static_assert(FragN % BlockN == 0u, "FragN must be a multiple of BlockN");
        static_assert(FragK % BlockK == 0u, "FragK must be a multiple of BlockK");

    private:

        template <typename VecTA, typename VecTB, typename VecTC>
        ROCWMMA_DEVICE static inline decltype(auto) exec_row_major(VecTA&& a, VecTB&& b, VecTC&& accum)
        {
            // Block-wise decomposition of the fragment size for matrix-matrix multiply
            auto a_frag = PreOps::execA(a);
            auto b_frag = PreOps::execB(b);
            auto c_frag = PreOps::execC(accum);

            // "Row-major" accumulation over the N-dimension blocks first.
            // Pseudo code here, but we would basically iterate over the blocks in row-major order
            for(uint32_t bm = 0u; bm < BlocksM; ++bm)
            {
                for(uint32_t bn = 0u; bn < BlocksN; ++bn)
                {
                    c_frag[bm][bn] = BlockWiseMma::exec(c_frag[bm][bn], a_frag[bm], b_frag[bn]);
                }
            }

            auto d_frag = PostOps::execD(c_frag);
            return d_frag;
        };

        template <typename VecTA, typename VecTB, typename VecTC>
        ROCWMMA_DEVICE static inline decltype(auto) exec_col_major(VecTA&& a, VecTB&& b, VecTC&& accum);

    public:

        template <typename VecTA, typename VecTB, typename VecTC>
        ROCWMMA_DEVICE static inline decltype(auto) exec(VecTA&& a, VecTB&& b, VecTC& accum);
    };

} // namespace ck::tile::core::arch
