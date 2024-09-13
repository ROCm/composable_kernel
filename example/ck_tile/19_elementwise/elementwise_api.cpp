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

#define DISPATCH_ELEMENTWISE_CAST(d_type_, s_type_, byte_per_issue_, chunks_)                      \
    using src_t = s_type_;                                                                         \
    using dst_t = d_type_;                                                                         \
    using u_fun = typename impl::Cast;                                                             \
    using problem =                                                                                \
        ck_tile::ElementwiseUnaryWarpPerRowProblem<src_t, dst_t, u_fun, byte_per_issue_, chunks_>; \
    using pipeline = ck_tile::ElementwiseUnaryipeline<problem>;                                    \
    using kernel   = ck_tile::ElementwiseUnaryKernel<pipeline>;                                    \
                                                                                                   \
    auto kargs            = kernel::MakeKargs(a);                                                  \
    const dim3 grids      = kernel::GridSize(a);                                                   \
    constexpr dim3 blocks = kernel::BlockSize();                                                   \
                                                                                                   \
    float ave_time = ck_tile::launch_kernel(                                                       \
        s, ck_tile::make_kernel<blocks.x, 1>(kernel{}, grids, blocks, 0, kargs));                  \
    return ave_time;

float elementwise(elementwise_trait t, elementwise_kargs a, ck_tile::stream_config s)
{
    float rtn = -1;
    if(t.op == "cast")
    {
        if(t.output_type == "fp32" && t.input_type == "fp16")
        {
            DISPATCH_ELEMENTWISE_CAST(float, ck_tile::fp16_t, sizeof(ck_tile::fp16_t), 8)
        }
        else if(t.output_type == "fp16" && t.input_type == "fp32")
        {
            DISPATCH_ELEMENTWISE_CAST(ck_tile::fp16_t, float, sizeof(float), 8)
        }
    }
    return rtn;
}
