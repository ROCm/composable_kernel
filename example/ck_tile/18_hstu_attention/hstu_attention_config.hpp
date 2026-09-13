// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// can be used on both host and kernel building
#if defined(BUILD_HSTU_FOR_GFX125)
#define __hstu_gfx125__ 1
#endif

// can be used on both host and kernel building
#if defined(BUILD_HSTU_FOR_GFX95)
#define __hstu_gfx95__ 1
#endif

// can be used on both host and kernel building
#if defined(BUILD_HSTU_FOR_GFX94)
#define __hstu_gfx94__ 1
#endif

#if defined(BUILD_HSTU_FOR_GFX95) || defined(BUILD_HSTU_FOR_GFX125)
#define HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE 1
#else
#define HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE 0
#endif
