// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/gemm/kernel/wavelet_utils.hpp"
#include <hip/hip_runtime.h>
#include <type_traits>

namespace ck_tile {
namespace dispatcher {
namespace backends {

// True for pipelines that declare skipCheckValidLaunchParams (the TDM pipelines).
// TDM clips each A/B box to the tensor's real extent; TdmEpilogue clips C too,
// whereas a non-TDM epilogue (CShuffle on comp_tdm_v3) still requires C tile
// divisibility when unpadded (UniversalGemmKernel::IsCTensorSupported).
template <typename Pipeline>
using pipeline_skips_launch_checks = std::bool_constant<ck_tile::pipeline_is_tdm_v<Pipeline>>;

// True for pipelines that only run on gfx125. This is ck_tile::pipeline_requires_gfx125_v, the
// trait UniversalGemmKernel::IsSupportedArgument gates on, so selection and launch cannot disagree.
template <typename Pipeline>
inline constexpr bool pipeline_requires_gfx125 = ck_tile::pipeline_requires_gfx125_v<Pipeline>;

// Same answer as ck_tile::is_gfx125_supported(), which UniversalGemmKernel::IsSupportedArgument
// uses, so selection and launch agree. supports() runs for every registry candidate on every
// select, and is_gfx125_supported() queries the device properties several times, so the answer is
// cached per thread for the current device; re-checking the device id keeps hipSetDevice safe.
inline bool current_device_is_gfx125()
{
    thread_local int cached_device = -1;
    thread_local bool cached_value = false;
    int device                     = -1;
    if(hipGetDevice(&device) != hipSuccess)
        return false;
    if(device != cached_device)
    {
        cached_value  = ck_tile::is_gfx125_supported();
        cached_device = device;
    }
    return cached_value;
}

// supports() gate shared by the generated backends: false when the pipeline needs gfx125 and the
// current device is not one, so first-fit falls through to a kernel valid for this GPU instead
// of throwing from IsSupportedArgument at launch. Other pipelines never query the device.
template <typename Pipeline>
inline bool device_supports_pipeline()
{
    if constexpr(pipeline_requires_gfx125<Pipeline>)
        return current_device_is_gfx125();
    else
        return true;
}

// Kernel-level form for TileKernelInstance. A Tile Engine SelectedKernel declares its GemmPipeline
// inside launch(), so it publishes kRequiresGfx125 at struct scope instead, and launch()
// static_asserts that flag against ck_tile::pipeline_requires_gfx125_v<GemmPipeline>. A kernel that
// exposes GemmPipeline is gated on the trait directly; a kernel with neither is left to its own
// IsSupportedArgument.
template <typename Kernel, typename = void>
struct kernel_gemm_pipeline
{
    using type = void;
};
template <typename Kernel>
struct kernel_gemm_pipeline<Kernel, std::void_t<typename Kernel::GemmPipeline>>
{
    using type = typename Kernel::GemmPipeline;
};

template <typename Kernel, typename = void>
struct kernel_declares_gfx125_requirement : std::false_type
{
};
template <typename Kernel>
struct kernel_declares_gfx125_requirement<Kernel, std::void_t<decltype(Kernel::kRequiresGfx125)>>
    : std::true_type
{
};

template <typename Kernel>
inline bool device_supports_kernel()
{
    using Pipeline = typename kernel_gemm_pipeline<Kernel>::type;
    if constexpr(!std::is_void_v<Pipeline>)
        return device_supports_pipeline<Pipeline>();
    else if constexpr(kernel_declares_gfx125_requirement<Kernel>::value)
    {
        if constexpr(Kernel::kRequiresGfx125)
            return current_device_is_gfx125();
        else
            return true;
    }
    else
        return true;
}

} // namespace backends
} // namespace dispatcher
} // namespace ck_tile
