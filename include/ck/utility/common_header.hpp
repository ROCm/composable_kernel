// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/utility/static_assert.hpp"
#include "ck/utility/remove_cvref.hpp"
#include "ck/utility/is_static.hpp"
#include "ck/utility/bit_cast.hpp"
#include "ck/utility/print.hpp"
#include "ck/utility/array.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/utility/tuple_helper.hpp"
#include "ck/utility/map.hpp"
#include "ck/utility/container_helper.hpp"
#include "ck/utility/statically_indexed_array.hpp"
#include "ck/utility/multi_index.hpp"
#include "ck/utility/data_type.hpp"
#include "ck/utility/functional.hpp"
#include "ck/utility/functional2.hpp"
#include "ck/utility/functional3.hpp"
#include "ck/utility/functional4.hpp"
#include "ck/utility/enable_if.hpp"
#include "ck/utility/ignore.hpp"
#include "ck/utility/integral_constant.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/utility/math_ext.hpp"
#include "ck/utility/number.hpp"
#include "ck/utility/tuple_of_sequence_to_array_of_array.hpp"
#include "ck/utility/macro_func_array_to_sequence.hpp"
#include "ck/utility/macro_func_array_of_array_to_tuple_of_sequence.hpp"
#include "ck/utility/type.hpp"
#include "ck/utility/type_convert.hpp"
#include "ck/utility/magic_division.hpp"
#include "ck/utility/c_style_pointer_cast.hpp"
#include "ck/utility/transpose_vectors.hpp"
#include "ck/utility/inner_product.hpp"
#include "ck/utility/thread_group.hpp"
#include "ck/utility/meta_data_buffer.hpp"
#include "ck/utility/debug.hpp"

#include "ck/utility/amd_buffer_addressing.hpp"
#include "ck/utility/amd_wave_read_first_lane.hpp"
#include "ck/utility/amd_warp_shuffle.hpp"
#include "ck/utility/generic_memory_space_atomic.hpp"
#include "ck/utility/get_id.hpp"
#include "ck/utility/thread_group.hpp"
#include "ck/utility/synchronization.hpp"
#include "ck/utility/amd_address_space.hpp"
#include "ck/utility/static_buffer.hpp"
#include "ck/utility/dynamic_buffer.hpp"

// TODO: remove this
#if CK_USE_AMD_INLINE_ASM
#include "ck/utility/amd_inline_asm.hpp"
#endif

#ifdef CK_USE_AMD_MFMA
#include "ck/utility/amd_xdlops.hpp"
#endif
