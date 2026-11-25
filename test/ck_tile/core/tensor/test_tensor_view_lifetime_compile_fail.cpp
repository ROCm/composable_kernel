// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Each CK_TENSOR_VIEW_LIFETIME_FAIL_* case binds a reference into the tensor view that a tile
// window returns by value; the ctest entries expect -Wdangling to reject it.

#include "ck_tile/core.hpp"

__global__ void bind_reference_into_temporary_view(float* p_data)
{
    using namespace ck_tile;

    const auto view = make_naive_tensor_view<address_space_enum::global>(
        p_data, make_tuple(number<16>{}, number<16>{}), make_tuple(number<16>{}, number<1>{}));
    const auto window = make_tile_window(view, make_tuple(number<16>{}, number<16>{}), {0, 0});

#if defined(CK_TENSOR_VIEW_LIFETIME_FAIL_DESCRIPTOR)
    const auto& descriptor = window.get_bottom_tensor_view().get_tensor_descriptor();
    ignore                 = descriptor;
#elif defined(CK_TENSOR_VIEW_LIFETIME_FAIL_BUFFER_VIEW)
    const auto& buffer_view = window.get_bottom_tensor_view().get_buffer_view();
    ignore                  = buffer_view;
#elif defined(CK_TENSOR_VIEW_LIFETIME_FAIL_CONST_BUFFER_VIEW)
    using View = remove_cvref_t<decltype(view)>;
    const auto& const_buffer_view =
        static_cast<const View>(window.get_bottom_tensor_view()).get_buffer_view();
    ignore = const_buffer_view;
#else
#error "Define one of CK_TENSOR_VIEW_LIFETIME_FAIL_{DESCRIPTOR,BUFFER_VIEW,CONST_BUFFER_VIEW}"
#endif
}
