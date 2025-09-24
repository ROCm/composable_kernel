#pragma once
#include <cstdint>
namespace ck_tile {
    enum struct address_space_enum : std::uint16_t {
        generic = 0,
        global,
        lds,
        sgpr,
        constant,
        vgpr
    };
}
