#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct SinkhornKnoppArgs
{
    const void* p_x;
    const index_t n;
    int max_iterations;
};

} // namespace ck_tile