// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

namespace ck_tile {

template <typename XDataType>
struct Docs
{
    
    template <typename InputShape>
    CK_TILE_DEVICE void operator()(XDataType* p_x,
                                   InputShape input_shape) const
    {

    if(threadIdx.x == 0){

        auto buffer_view = make_buffer_view<address_space_enum::global>(p_x, 10);

        // buffer_view.template set<int>(0, 0, false, 99);

        // auto value = buffer_view.template get<XDataType>(0, 0, true);
        // printf("Element: %d\n", value.get(0));

        using int2 = ext_vector_t<int, 2>;
        buffer_view.template set<int2>(0, 0, false, {});


        
        auto vector = buffer_view.template get<int2>(0, 0, true);
        printf("Vector read (2 elements from index 1): [%d, %d]\n", 
            vector.get(0), vector.get(1));



    }
    }
};

} // namespace ck_tile
