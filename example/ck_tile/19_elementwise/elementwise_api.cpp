#include "elementwise_api.hpp"

namespace impl {
struct Cast
{
    template <typename DstType, typename SrcType>
    CK_TILE_HOST_DEVICE void operator()(DstType& y, const SrcType& x) const
    {
        y = ck_tile::type_convert<DstType>(x);
    };
};
} // namespace impl

#define DISPATCH_E_CAST_(d_type_, s_type_, byte_per_issue_, chunks_, bs_)                      \
    using src_t   = s_type_;                                                                   \
    using dst_t   = d_type_;                                                                   \
    using u_fun   = typename impl::Cast;                                                       \
    using problem = ck_tile::                                                                  \
        ElementwiseUnaryWarpPerRowProblem<src_t, dst_t, u_fun, byte_per_issue_, chunks_, bs_>; \
    using pipeline = ck_tile::ElementwiseUnaryipeline<problem>;                                \
    using kernel   = ck_tile::ElementwiseUnaryKernel<pipeline>;                                \
                                                                                               \
    auto kargs            = kernel::MakeKargs(a);                                              \
    const dim3 grids      = kernel::GridSize(a);                                               \
    constexpr dim3 blocks = kernel::BlockSize();                                               \
                                                                                               \
    float ave_time = ck_tile::launch_kernel(                                                   \
        s,                                                                                     \
        ck_tile::make_kernel<blocks.x, 1>(                                                     \
            kernel{}, grids, blocks, 0, kargs.p_input, kargs.p_output, kargs.num_pixels));     \
    return ave_time;

float elementwise(elementwise_trait t, elementwise_kargs a, ck_tile::stream_config s)
{
    float rtn = -1;
    if(t.op == "cast")
    {
        if(t.output_type == "fp32" && t.input_type == "fp16")
        {
            constexpr int eb = sizeof(ck_tile::fp16_t);
            if(a.num_pixels < (static_cast<uint64_t>(t.num_cu) * 64))
            {
                DISPATCH_E_CAST_(float, ck_tile::fp16_t, 1 * eb, 1, 64)
            }
            else if(a.num_pixels < (static_cast<uint64_t>(t.num_cu) * 128))
            {
                DISPATCH_E_CAST_(float, ck_tile::fp16_t, 1 * eb, 1, 128)
            }
            else if(a.num_pixels < (static_cast<uint64_t>(t.num_cu) * 256 * 3))
            {
                DISPATCH_E_CAST_(float, ck_tile::fp16_t, 1 * eb, 1, 256)
            }
            else if(a.num_pixels % 4 == 0)
            {
                if(a.num_pixels < (static_cast<uint64_t>(t.num_cu) * 256 * 4 * 8))
                {
                    DISPATCH_E_CAST_(float, ck_tile::fp16_t, 4 * eb, 1, 256)
                }
                else
                {
                    DISPATCH_E_CAST_(float, ck_tile::fp16_t, 4 * eb, 8, 256)
                }
            }
            else
            {
                DISPATCH_E_CAST_(float, ck_tile::fp16_t, 1 * eb, 1, 256)
            }
        }
        else if(t.output_type == "fp16" && t.input_type == "fp32")
        {
            constexpr int eb = sizeof(float);
            if(a.num_pixels < (static_cast<uint64_t>(t.num_cu) * 64))
            {
                DISPATCH_E_CAST_(ck_tile::fp16_t, float, 1 * eb, 1, 64)
            }
            else if(a.num_pixels < (static_cast<uint64_t>(t.num_cu) * 128))
            {
                DISPATCH_E_CAST_(ck_tile::fp16_t, float, 1 * eb, 1, 128)
            }
            else if(a.num_pixels < (static_cast<uint64_t>(t.num_cu) * 256 * 3))
            {
                DISPATCH_E_CAST_(ck_tile::fp16_t, float, 1 * eb, 1, 256)
            }
            else if(a.num_pixels % 4 == 0)
            {
                if(a.num_pixels < (static_cast<uint64_t>(t.num_cu) * 256 * 4 * 8))
                {
                    DISPATCH_E_CAST_(ck_tile::fp16_t, float, 4 * eb, 1, 256)
                }
                else
                {
                    DISPATCH_E_CAST_(ck_tile::fp16_t, float, 4 * eb, 8, 256)
                }
            }
            else
            {
                DISPATCH_E_CAST_(ck_tile::fp16_t, float, 1 * eb, 1, 256)
            }
        }
    }
    return rtn;
}
