#pragma once
#include "config.hpp"
#include "array.hpp"
#include "container_helper.hpp"
#include "statically_indexed_array.hpp"
#include "container_element_picker.hpp"
#include "multi_index.hpp"
#include "data_type.hpp"
#include "data_type_enum.hpp"
#include "data_type_enum_helper.hpp"
#include "functional.hpp"
#include "functional2.hpp"
#include "functional3.hpp"
#include "functional4.hpp"
#include "enable_if.hpp"
#include "ignore.hpp"
#include "integral_constant.hpp"
#include "math.hpp"
#include "number.hpp"
#include "sequence.hpp"
#include "sequence_helper.hpp"
#include "tuple.hpp"
#include "tuple_helper.hpp"
#include "type.hpp"
#include "magic_division.hpp"
#include "c_style_pointer_cast.hpp"
#include "is_known_at_compile_time.hpp"
#include "transpose_vectors.hpp"
#include "inner_product.hpp"
#include "element_wise_operation.hpp"
#include "thread_group.hpp"
#include "debug.hpp"

#include "amd_buffer_addressing.hpp"
#include "generic_memory_space_atomic.hpp"
#include "get_id.hpp"
#include "synchronization.hpp"
#include "amd_address_space.hpp"
#include "static_buffer.hpp"
#include "dynamic_buffer.hpp"

// TODO: remove this
#if CK_USE_AMD_INLINE_ASM
#include "amd_inline_asm.hpp"
#endif

#ifdef CK_USE_AMD_MFMA
#include "amd_xdlops.hpp"
#endif

#define USEING_STATIC_KERNEL 1

#define MNKB_0_8 0
#define MNKB_1_4 0
#define MNKB_2_8 0
#define MNKB_3_5 0

#define MNKB_4_5 0
#define MNKB_5_5 0

#if MNKB_0_8
#define M_matrix 16
#define N_matrix 1152
#define K_matrix 5120
#define K_batch 8
#elif MNKB_1_4
#define M_matrix 16
#define N_matrix 5120
#define K_matrix 384
#define K_batch 4
#elif MNKB_2_8
#define M_matrix 16
#define N_matrix 1280
#define K_matrix 5120
#define K_batch 8
#elif MNKB_3_5
#define M_matrix 16
#define N_matrix 5120
#define K_matrix 1280
#define K_batch 5
#elif MNKB_4_5
#define M_matrix 16
#define N_matrix 4096
#define K_matrix 12800
#define K_batch 5
#elif MNKB_5_5
#define M_matrix 16
#define N_matrix 4096
#define K_matrix 12800
#define K_batch 5
#endif
