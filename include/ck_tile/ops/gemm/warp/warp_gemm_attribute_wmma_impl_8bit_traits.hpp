// int8 specialization - GFX11
template <>
struct WmmaTraits<gfx11_t, int8_t, int8_t, int32_t, 16, 16, 16>
    : WmmaTraitsBase<gfx11_t, int8_t, int8_t, int32_t>
{
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx11__
        return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, // neg_a
                                                          bit_cast<int32x4_t>(a_vec),
                                                          true, // neg_b
                                                          bit_cast<int32x4_t>(b_vec),
                                                          bit_cast<int32x8_t>(c_vec),
                                                          clamp);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0};
#endif
    }
};
