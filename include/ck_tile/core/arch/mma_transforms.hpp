// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "mfma/mfma.hpp"
#include "wmma/wmma.hpp"

#include "mma_traits.hpp"

namespace ck::tile::core::arch
{
    // Default no-op transform policy
    struct PassThroughTransform
    {
        template<typename VecType>
        CK_TILE_DEVICE static decltype(auto) exec(VecType const& v)
        {
            return v;
        }
    };

    // Builds a transform policy around a given MMA operation for inputs
    // Gives access to MmaOp for type traits
    template<typename MmaOp, 
             typename TransformC = PassThroughTransform,
             typename TransformA = PassThroughTransform,
             typename TransformB = PassThroughTransform>
    struct PreMmaTransform
    {
        using Traits = mma_traits<MmaOp>;

        CK_TILE_DEVICE static decltype(auto) execC(typename Traits::CVecType const& c)
        {
            return TransformC::exec(c);
        }

        CK_TILE_DEVICE static decltype(auto) execA(typename Traits::AVecType const& a)
        {
            return TransformA::exec(a);
        }

        CK_TILE_DEVICE static decltype(auto) execB(typename Traits::BVecType const& b)
        {
            return TransformB::exec(b);
        }
    };

    // Builds a transform policy around a given MMA operation for outputs
    // Gives access to MmaOp for type traits
    template<typename MmaOp, 
             typename TransformD = PassThroughTransform>
    struct PostMmaTransform
    {
        using Traits = mma_traits<MmaOp>;

        CK_TILE_DEVICE static decltype(auto) execD(typename Traits::DVecType const& d)
        {
            return TransformD::exec(d);
        }
    };

    namespace detail
    {
        // Implements a default TransformSelector that can be specialized for specific MmaOps
        // or other conditions (e.g., architecture)
        template<typename MmaOp, typename Enable = void>
        struct TransformSelector
        {
            using Pre = PreMmaTransform<MmaOp, 
                                        PassThroughTransform,
                                        PassThroughTransform,
                                        PassThroughTransform>;

            using Post = PostMmaTransform<MmaOp,
                                          PassThroughTransform>;
        };

        // Example of a transform selector specialization for WMMA ops on GFX11
        // that applies specific pre- and post-MMA transforms
        template<typename MmaOp>
        struct TransformSelector<MmaOp, std::enable_if_t<mma_traits<MmaOp>::IsWmma && CK_TILE_ARCH_GFX11>>
        {
            using Pre = PreMmaTransform<MmaOp,
                                        wmma::DuplicateTransformGfx11,
                                        wmma::DuplicateTransformGfx11, 
                                        wmma::PadTransformGfx11>;

            using Post = PostMmaTransform<MmaOp,
                                          wmma::UnpadTransformGfx11>;
        };

        // Can implement more TransformSelector specializations here for other architectures or situations...

    } // namespace detail

    // Assemble a front-end transform selector 
    template<typename MmaOp>
    struct TransformSelector : public detail::TransformSelector<MmaOp>
    {
    };

} // namespace ck::tile::core::arch
