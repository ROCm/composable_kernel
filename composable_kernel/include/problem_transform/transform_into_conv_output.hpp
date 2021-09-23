#ifndef CK_TRANSFORM_INTO_CONV_OUTPUT_HPP
#define CK_TRANSFORM_INTO_CONV_OUTPUT_HPP

#include "multi_index_transform_helper.hpp"
#include "common_header.hpp"
#include "tensor_descriptor.hpp"

namespace ck {

/*
 * This functors are used to fuse convolution with some other operators. For example, 
 * in order to fuse conv + depth2space, the output of depth2space has to be transformed
 * into the output of convolution.
 *
 * TODO: Use universal reference parameter in functor operators?
 */

template < index_t BlockSize>
struct TransformDepth2SpaceToConvolution_nhwc;

struct NoTransform
{
    template <typename... DescArgs>
    __host__ __device__ constexpr auto operator () (
    const TensorDescriptor<DescArgs...>& conv_out)
    {
        return conv_out;
    }
};

template <>
struct TransformDepth2SpaceToConvolution_nhwc<1>
{
    template <typename... DescArgs>
    __host__ __device__ constexpr auto operator () (
    const TensorDescriptor<DescArgs...>& conv_out)
    {
        return conv_out;
    }

};

template <index_t BlockSize>
struct TransformDepth2SpaceToConvolution_nhwc
{
    template <typename... DescArgs>
    __host__ __device__ constexpr auto operator () (
    const TensorDescriptor<DescArgs...>& depth2space_n_hobs_wobs_c_desc)
    {
        constexpr auto I0 = Number<0>{};
        constexpr auto I1 = Number<1>{};
        constexpr auto I2 = Number<2>{};
        constexpr auto I3 = Number<3>{};
    
        const auto N = depth2space_n_hobs_wobs_c_desc.GetLength(I0);
        const auto HoBs = depth2space_n_hobs_wobs_c_desc.GetLength(I1);
        const auto WoBs = depth2space_n_hobs_wobs_c_desc.GetLength(I2);
        const auto C = depth2space_n_hobs_wobs_c_desc.GetLength(I3);
        assert(HoBs % BlockSize == 0);
        assert(WoBs % BlockSize == 0);
        const auto Ho = HoBs / BlockSize;
        const auto Wo = WoBs / BlockSize;

        const auto depth2space_n_ho_wo_b0_b1_c_desc = transform_tensor_descriptor(
            depth2space_n_hobs_wobs_c_desc,
            make_tuple(make_pass_through_transform(N),
                       make_unmerge_transform(make_tuple(Ho, BlockSize)),
                       make_unmerge_transform(make_tuple(Wo, BlockSize)),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2, 4>{}, Sequence<5>{}));

        const auto conv_out_n_ho_wo_k_desc = transform_tensor_descriptor(
            depth2space_n_ho_wo_b0_b1_c_desc,
            make_tuple(make_pass_through_transform(N),
                       make_pass_through_transform(Ho),
                       make_pass_through_transform(Wo),
                       make_merge_transform(make_tuple(BlockSize, BlockSize, C))),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3, 4, 5>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        assert(conv_out_n_ho_wo_k_desc.GetLength(I0) == N);
        assert(conv_out_n_ho_wo_k_desc.GetLength(I1)*BlockSize == HoBs);
        assert(conv_out_n_ho_wo_k_desc.GetLength(I2)*BlockSize == WoBs);
        assert(conv_out_n_ho_wo_k_desc.GetLength(I3) == C*BlockSize*BlockSize);

        return conv_out_n_ho_wo_k_desc;
    }
};

} // namespace ck
#endif

