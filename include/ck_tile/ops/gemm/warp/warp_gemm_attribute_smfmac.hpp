#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_smfmac_impl.hpp"

namespace ck_tile {

template <typename WarpGemmAttributeSmfmacImpl_>
struct WarpGemmAttributeSmfmac
{
    using Impl = remove_cvref_t<WarpGemmAttributeSmfmacImpl_>;

    using ADataType   = typename Impl::ADataType;
    using BDataType   = typename Impl::BDataType;
    using IdxDataType = typename Impl::IdxDataType;
    using CDataType   = typename Impl::CDataType;

    using AVecType = typename Impl::AVecType;
    using BVecType = typename Impl::BVecType;
    using CVecType = typename Impl::CVecType;

    static constexpr index_t kM          = Impl::kM;
    static constexpr index_t kN          = Impl::kN;
    static constexpr index_t kK          = Impl::kK;
    static constexpr index_t kKPerThread = Impl::kABKPerLane;

    CK_TILE_HOST_DEVICE static constexpr auto get_num_of_access() { return 1; }

    static_assert(Impl::kAMBlock == 1 && Impl::kBNBlock == 1,
                  "Multi-block WarpGemmAttributeMfmaImpl is not supported");

    using AWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kAMLane>, sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using BWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kBNLane>, sequence<Impl::kABKLane, Impl::kABKPerLane>>,
        tuple<sequence<2, 1>>,
        tuple<sequence<0, 0>>,
        sequence<2>,
        sequence<1>>;

    using CWarpDstrEncoding = tile_distribution_encoding<
        sequence<>,
        tuple<sequence<Impl::kCM0PerLane, Impl::kCMLane, Impl::kCM1PerLane>,
              sequence<Impl::kCNLane>>,
        tuple<sequence<1, 2>>,
        tuple<sequence<1, 0>>,
        sequence<1, 1>,
        sequence<0, 2>>;

    // c_vec += a_vec * b_vec[idx]
    template <bool post_nop_ = false>
    CK_TILE_DEVICE void operator()(CVecType& c_vec,
                                   const AVecType& a_vec,
                                   const BVecType& b_vec,
                                   const int32_t& idx,
                                   bool_constant<post_nop_> = {}) const
    {
        Impl{}(c_vec, a_vec, b_vec, idx, bool_constant<post_nop_>{});
    }
};
} // namespace ck_tile
