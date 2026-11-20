// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_default_policy.hpp"

namespace ck_tile {

/**
 * @brief Policy of GemmPipelineAgBgCrCompTDMRunAhead.
 *
 * The wave-specialised TDM policy, under which one wave's transfer moves an operand's whole
 * block tile, plus the two knobs of the ring the math waves fill for themselves.
 *
 * @tparam NumSlots_ LDS buffers in the ring.
 * @tparam RunAhead_ K steps the fills run ahead of the math, i.e. transfers each filling wave
 *                   keeps in flight per operand. At most NumSlots_ - 1: the fill for step
 *                   k + RunAhead_ reuses the slot step k + RunAhead_ - NumSlots_ read, and only
 *                   steps up to k - 1 are known to be drained when it is issued.
 */
template <index_t NumSlots_ = 4, index_t RunAhead_ = NumSlots_ - 1>
struct GemmPipelineAgBgCrCompTDMRunAheadPolicy : public GemmPipelineAgBgCrCompTDMDefaultPolicy<true>
{
    static constexpr index_t NumSlots = NumSlots_;
    static constexpr index_t RunAhead = RunAhead_;

    static_assert(NumSlots >= 2, "the ring needs a slot to fill while another is read");
    static_assert(RunAhead >= 1 && RunAhead < NumSlots,
                  "RunAhead must be in [1, NumSlots): a fill may only reuse a slot whose last "
                  "read is at least one step behind the step being consumed");
};

} // namespace ck_tile
