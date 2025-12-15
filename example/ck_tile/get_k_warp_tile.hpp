#ifndef GET_K_WARP_TILE_HPP
#define GET_K_WARP_TILE_HPP

#include <type_traits> 

namespace ck_tile
{
    using index_t = int;

    struct fp8_t {};
    struct bf8_t {};
}
template <typename PrecType, int M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
{
#if defined(CK_GFX950_SUPPORT)
    constexpr bool is_8bit_float =
        std::is_same_v<PrecType, ck_tile::fp8_t> || std::is_same_v<PrecType, ck_tile::bf8_t>;

    if constexpr (M_Warp_Tile == 32)
        return is_8bit_float ? 64 : 16;
    else
        return is_8bit_float ? 128 : 32;
#else
    if constexpr (M_Warp_Tile == 32)
        return 16;
    else
        return 32;
#endif
}

#endif 
