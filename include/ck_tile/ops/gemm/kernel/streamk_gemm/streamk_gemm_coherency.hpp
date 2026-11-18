// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core/arch/arch.hpp"
namespace ck_tile {

// BUFFER_COHERENCE: partial-tile stores and the completion flags. LOAD_COHERENCE: partial-tile
// loads, which read another workgroup's stores and so need device scope where the default does
// not reach the whole device (gfx1250, whose XCCs have separate L2s).
template <typename CompilerTarget, typename Enabler = void>
struct StreamKCoherency
{
    static constexpr amd_buffer_coherence_enum BUFFER_COHERENCE =
        amd_buffer_coherence_enum::coherence_default;
    static constexpr amd_buffer_coherence_enum LOAD_COHERENCE =
        amd_buffer_coherence_enum::coherence_default;
};

template <typename CompilerTarget>
struct StreamKCoherency<CompilerTarget,
                        core::arch::enable_if_target_id_t<CompilerTarget,
                                                          core::arch::amdgcn_target_id::GFX942,
                                                          core::arch::amdgcn_target_id::GFX950>>
{
    static constexpr amd_buffer_coherence_enum BUFFER_COHERENCE =
        amd_buffer_coherence_enum::SYSTEM_NT0;
    static constexpr amd_buffer_coherence_enum LOAD_COHERENCE =
        amd_buffer_coherence_enum::coherence_default;
};

template <typename CompilerTarget>
struct StreamKCoherency<CompilerTarget,
                        core::arch::enable_if_target_id_t<CompilerTarget,
                                                          core::arch::amdgcn_target_id::GFX908,
                                                          core::arch::amdgcn_target_id::GFX90A>>
{
    static constexpr amd_buffer_coherence_enum BUFFER_COHERENCE =
        amd_buffer_coherence_enum::glc_slc;
    static constexpr amd_buffer_coherence_enum LOAD_COHERENCE =
        amd_buffer_coherence_enum::coherence_default;
};

template <typename CompilerTarget>
struct StreamKCoherency<
    CompilerTarget,
    core::arch::enable_if_target_id_t<CompilerTarget,
                                      core::arch::amdgcn_target_id::GFX1200,
                                      core::arch::amdgcn_target_id::GFX1201,
                                      core::arch::amdgcn_target_id::GFX1250,
                                      core::arch::amdgcn_target_id::GFX12_GENERIC>>
{
    static constexpr amd_buffer_coherence_enum BUFFER_COHERENCE = amd_buffer_coherence_enum::DEVICE;
    // Only gfx1250 has been shown to need device-scope partial loads; the rest of gfx12 keeps
    // the default until it is validated there.
    static constexpr amd_buffer_coherence_enum LOAD_COHERENCE =
        core::arch::is_target_id_any_of<CompilerTarget, core::arch::amdgcn_target_id::GFX1250>()
            ? amd_buffer_coherence_enum::DEVICE
            : amd_buffer_coherence_enum::coherence_default;
};

template <typename CompilerTarget>
struct StreamKCoherency<CompilerTarget,
                        std::enable_if_t<core::arch::is_target_family_gfx11<CompilerTarget>() ||
                                         core::arch::is_target_family_gfx117<CompilerTarget>()>>
{
    static constexpr amd_buffer_coherence_enum BUFFER_COHERENCE =
        amd_buffer_coherence_enum::glc_dlc;
    static constexpr amd_buffer_coherence_enum LOAD_COHERENCE =
        amd_buffer_coherence_enum::coherence_default;
};

} // namespace ck_tile
