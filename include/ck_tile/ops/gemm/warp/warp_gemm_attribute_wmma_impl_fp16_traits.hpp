// GFX11 specialization
template <>
struct WmmaTraits<gfx11_t, fp16_t, fp16_t, float, 16, 16, 16>
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 16>;
    using BVecType = ext_vector_t<fp16_t, 16>;
    using CVecType = ext_vector_t<float, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kRepeat      = 2;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 1;
    static constexpr index_t kABKLane     = 1;
    static constexpr index_t kABK1PerLane = 16;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 8;
    static constexpr index_t kCM1PerLane = 1;

    using kABPs2RHssMajor = sequence<0, 2, 1>;
    using kABPs2RHssMinor = sequence<0, 1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssTransMajor = sequence<2, 1>;
    using kCPs2RHssTransMinor = sequence<1, 0>;
    using kCYs2RHsTransMajor  = sequence<2, 2>;
    using kCYs2RHsTransMinor  = sequence<0, 2>;

    // Architecture-specific WMMA intrinsic
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx11__
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};

// GFX12 specialization
template <>
struct WmmaTraits<gfx12_t, fp16_t, fp16_t, float, 16, 16, 16>
{
    using ADataType = fp16_t;
    using BDataType = fp16_t;
    using CDataType = float;

    using AVecType = ext_vector_t<fp16_t, 8>;
    using BVecType = ext_vector_t<fp16_t, 8>;
    using CVecType = ext_vector_t<float, 8>;

    static constexpr index_t kM = 16;
    static constexpr index_t kN = 16;
    static constexpr index_t kK = 16;

    static constexpr index_t kRepeat      = 1;
    static constexpr index_t kAMLane      = 16;
    static constexpr index_t kBNLane      = 16;
    static constexpr index_t kABK0PerLane = 2;
    static constexpr index_t kABKLane     = 2;
    static constexpr index_t kABK1PerLane = 4;

    static constexpr index_t kCMLane     = 2;
    static constexpr index_t kCNLane     = 16;
    static constexpr index_t kCM0PerLane = 1;
    static constexpr index_t kCM1PerLane = 8;

    using kABPs2RHssMajor = sequence<2, 1>;
    using kABPs2RHssMinor = sequence<1, 0>;
    using kABYs2RHsMajor  = sequence<2, 2>;
    using kABYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssMajor = sequence<1, 2>;
    using kCPs2RHssMinor = sequence<1, 0>;
    using kCYs2RHsMajor  = sequence<1, 1>;
    using kCYs2RHsMinor  = sequence<0, 2>;

    using kCPs2RHssTransMajor = sequence<2, 1>;
    using kCPs2RHssTransMinor = sequence<1, 0>;
    using kCYs2RHsTransMajor  = sequence<2, 2>;
    using kCYs2RHsTransMinor  = sequence<0, 2>;

    // Architecture-specific WMMA intrinsic
    template <bool clamp = false>
    CK_TILE_DEVICE static CVecType
    wmma_intrinsic(const AVecType& a_vec, const BVecType& b_vec, const CVecType& c_vec)
    {
#ifdef __gfx12__
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_vec, b_vec, c_vec);
#else
        ck_tile::ignore = a_vec;
        ck_tile::ignore = b_vec;
        ck_tile::ignore = c_vec;
        return CVecType{0.f};
#endif
    }
};
