// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp"

namespace ck_tile {

/**
 * @brief Policy of GemmPipelineAgBgCrCompTDMProducerConsumer.
 *
 * The wave-specialised TDM policy, under which one wave's transfer moves an operand's whole
 * block tile, plus the two knobs of the ring between loaders and compute. Both are meant to be
 * tuned on hardware.
 *
 * @tparam NumSlots_   LDS buffers in the ring, i.e. how many K steps the loaders may run ahead.
 *                     Two loaders need 3 * NumSlots named barriers, which caps it at 5.
 * @tparam PublishLag_ Transfers each loader keeps in flight beyond the one it publishes; below
 *                     NumSlots_. At NumSlots_ - 1, publishing a slot waits for the drain of the
 *                     one read just before it, a loader round trip on every compute step.
 */
template <index_t NumSlots_ = 4, index_t PublishLag_ = 0>
struct GemmPipelineAgBgCrCompTDMProducerConsumerPolicy
    : public GemmPipelineAgBgCrCompTDMDefaultPolicy<true>
{
    static constexpr index_t NumSlots   = NumSlots_;
    static constexpr index_t PublishLag = PublishLag_;
};

} // namespace ck_tile
