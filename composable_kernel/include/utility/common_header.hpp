#ifndef CK_COMMON_HEADER_HPP
#define CK_COMMON_HEADER_HPP

#include "config.hpp"
#include "utility.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "type.hpp"
#include "tuple.hpp"
#include "math.hpp"
#include "vector_type.hpp"
#include "sequence.hpp"
#include "sequence_helper.hpp"
#include "array.hpp"
#include "array_helper.hpp"
#include "functional.hpp"
#include "functional2.hpp"
#include "functional3.hpp"
#include "functional4.hpp"

#if CK_USE_AMD_INLINE_ASM
#include "amd_inline_asm.hpp"
#endif

#if CK_USE_AMD_INTRINSIC
#include "amd_intrinsic.hpp"
#endif

#endif
