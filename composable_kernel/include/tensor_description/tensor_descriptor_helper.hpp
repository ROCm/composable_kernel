#ifndef CK_TENSOR_DESCRIPTOR_HELPER_HPP
#define CK_TENSOR_DESCRIPTOR_HELPER_HPP

#include "common_header.hpp"
#include "tensor_descriptor.hpp"
#include "multi_index_transform_helper.hpp"

namespace ck {

/*
 * These functions create tensor descriptor at runtime. If they are not constexpr, you will
 * likely see usage of scratch memory during construction of these tensor descriptors. So
 * it's better to call these functions on host and then pass the constructed tensor descritpors
 * to GPU. If the tensor descritpors being constructed are constexpr, then you can call these
 * functions on GPU without worrying about scratch memory usage.
 */

#if CK_WORKAROUND_SWDEV_275126
template <typename Lengths, typename Strides, index_t I, typename AccOld>
__host__ __device__ constexpr auto calculate_element_space_size_impl(const Lengths& lengths,
                                                                     const Strides& strides,
                                                                     Number<I> i,
                                                                     AccOld acc_old)
{
    auto acc_new = acc_old + (lengths[i] - Number<1>{}) * strides[i];

    if constexpr(i.value < Lengths::Size() - 1)
    {
        return calculate_element_space_size_impl(lengths, strides, i + Number<1>{}, acc_new);
    }
    else
    {
        return acc_new;
    }
}
#endif

template <typename... Lengths,
          typename... Strides,
          typename enable_if<sizeof...(Lengths) == sizeof...(Strides), bool>::type = false>
__host__ __device__ constexpr auto make_naive_tensor_descriptor(const Tuple<Lengths...>& lengths,
                                                                const Tuple<Strides...>& strides)
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(make_embed_transform(lengths, strides));

    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

#if !CK_WORKAROUND_SWDEV_275126
    // rocm-4.1 compiler would crash for recursive labmda
    // recursive function for reduction
    auto f = [&](auto fs, auto i, auto acc_old) {
        auto acc_new = acc_old + (lengths[i] - Number<1>{}) * strides[i];

        if constexpr(i.value < N - 1)
        {
            return fs(fs, i + Number<1>{}, acc_new);
        }
        else
        {
            return acc_new;
        }
    };

    const auto element_space_size = f(f, Number<0>{}, Number<1>{});
#else
    const auto element_space_size =
        calculate_element_space_size_impl(lengths, strides, Number<0>{}, Number<1>{});
#endif

    return TensorDescriptor<remove_cv_t<decltype(transforms)>,
                            remove_cv_t<decltype(low_dim_hidden_idss)>,
                            remove_cv_t<decltype(up_dim_hidden_idss)>,
                            remove_cv_t<decltype(visible_dim_hidden_ids)>,
                            remove_cv_t<decltype(element_space_size)>>{transforms,
                                                                       element_space_size};
}

// Lengths... can be:
//   1) index_t, which is known at run-time
//   2) Number<>, which is known at compile-time
template <typename... Lengths>
__host__ __device__ constexpr auto
make_naive_tensor_descriptor_packed(const Tuple<Lengths...>& lengths)
{
    constexpr index_t N = sizeof...(Lengths);

    const auto transforms = make_tuple(make_unmerge_transform(lengths));

    constexpr auto low_dim_hidden_idss = make_tuple(Sequence<0>{});

    constexpr auto up_dim_hidden_idss =
        make_tuple(typename arithmetic_sequence_gen<1, N + 1, 1>::type{});

    constexpr auto visible_dim_hidden_ids = typename arithmetic_sequence_gen<1, N + 1, 1>::type{};

    const auto element_space_size = container_reduce(lengths, math::multiplies{}, Number<1>{});

    return TensorDescriptor<remove_cv_t<decltype(transforms)>,
                            remove_cv_t<decltype(low_dim_hidden_idss)>,
                            remove_cv_t<decltype(up_dim_hidden_idss)>,
                            remove_cv_t<decltype(visible_dim_hidden_ids)>,
                            remove_cv_t<decltype(element_space_size)>>{transforms,
                                                                       element_space_size};
}

template <typename... Lengths, typename Align>
__host__ __device__ constexpr auto
make_naive_tensor_descriptor_aligned(const Tuple<Lengths...>& lengths, Align align)
{
    constexpr auto I1 = Number<1>{};

    constexpr index_t N = sizeof...(Lengths);

    const auto stride_n_minus_2 = math::integer_least_multiple(lengths[Number<N - 1>{}], align);

    auto strides = generate_tuple(
        [&](auto i) {
            if constexpr(i.value == N - 1)
            {
                return I1;
            }
            else if constexpr(i.value == N - 2)
            {
                return Number<stride_n_minus_2>{};
            }
            else
            {
                return container_reduce(lengths,
                                        math::multiplies{},
                                        Number<stride_n_minus_2>{},
                                        i + I1,
                                        Number<N - 1>{},
                                        I1);
            }
        },
        Number<N>{});

    return make_naive_tensor_descriptor(lengths, strides);
}

template <index_t BlockSize, typename... Depth2Space>
__host__ __device__ constexpr auto transform_depth2space_to_convolution_nchw(
    const TensorDescriptor<Depth2Space...>& depth2space_n_c_hobs_wobs_desc)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = depth2space_n_c_hobs_wobs_desc.GetLength(I0);
    const auto C = depth2space_n_c_hobs_wobs_desc.GetLength(I1);
    const auto HoBs = depth2space_n_c_hobs_wobs_desc.GetLength(I2);
    const auto WoBs = depth2space_n_c_hobs_wobs_desc.GetLength(I3);
    assert(HoBs % / BlockSize == 0);
    assert(WoBs % / BlockSize == 0);
    const auto Ho = HoBs / BlockSize;
    const auto Wo = WoBs / BlockSize;

#define _depth2space_transform_ 2

#if _depth2space_transform_ == 0
    const auto depth2space_n_c_ho_b0_wo_b1_desc = transform_tensor_descriptor(
        depth2space_n_c_hobs_wobs_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_unmerge_transform(make_tuple(Ho, BlockSize)),
                   make_unmerge_transform(make_tuple(Wo, BlockSize))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto conv_out_n_k_ho_wo_desc = transform_tensor_descriptor(
        depth2space_out_n_c_ho_b0_wo_b1_desc,
        make_tuple(make_pass_through_transform(N),
                   make_merge_transform(make_tuple(C, BlockSize, BlockSize)),
                   make_pass_through_transform(Ho),
                   make_pass_through_transform(Wo)),
        make_tuple(Sequence<0>{}, Sequence<1, 3, 5>{}, Sequence<2>{}, Sequence<4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#elif _depth2space_transform_ == 1
    const auto depth2space_n_c_b0_ho_b1_wo_desc = transform_tensor_descriptor(
        depth2space_n_c_hobs_wobs_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_unmerge_transform(make_tuple(BlockSize, Ho)),
                   make_unmerge_transform(make_tuple(BlockSize, Wo))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 3>{}, Sequence<4, 5>{}));

    const auto conv_out_n_k_ho_wo_desc = transform_tensor_descriptor(
        depth2space_n_c_b0_ho_b1_wo_desc,
        make_tuple(make_pass_through_transform(N),
                   make_merge_transform(make_tuple(C, BlockSize, BlockSize)),
                   make_pass_through_transform(Ho),
                   make_pass_through_transform(Wo)),
        make_tuple(Sequence<0>{}, Sequence<1, 2, 4>{}, Sequence<3>{}, Sequence<5>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#elif _depth2space_transform_ == 2
    const auto depth2space_n_c_b0_b1_ho_wo_desc = transform_tensor_descriptor(
        depth2space_n_c_hobs_wobs_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(C),
                   make_unmerge_transform(make_tuple(BlockSize, Ho)),
                   make_unmerge_transform(make_tuple(BlockSize, Wo))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2, 4>{}, Sequence<3, 5>{}));

    const auto conv_out_n_k_ho_wo_desc = transform_tensor_descriptor(
        depth2space_n_c_b0_b1_ho_wo_desc,
        make_tuple(make_pass_through_transform(N),
                   make_merge_transform(make_tuple(C, BlockSize, BlockSize)),
                   make_pass_through_transform(Ho),
                   make_pass_through_transform(Wo)),
        make_tuple(Sequence<0>{}, Sequence<1, 2, 3>{}, Sequence<4>{}, Sequence<5>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#endif
    return conv_out_n_k_ho_wo_desc;
#undef _depth2space_transform_
}

template < index_t BlockSize, typename... Depth2Space>
__host__ __device__ constexpr auto transform_depth2space_to_convolution_nhwc(
    const TensorDescriptor<Depth2Space...>& depth2space_n_hobs_wobs_c_desc)
{
    constexpr auto I0 = Number<0>{};
    constexpr auto I1 = Number<1>{};
    constexpr auto I2 = Number<2>{};
    constexpr auto I3 = Number<3>{};

    const auto N = depth2space_n_hobs_wobs_c_desc.GetLength(I0);
    const auto HoBs = depth2space_n_hobs_wobs_c_desc.GetLength(I1);
    const auto WoBs = depth2space_n_hobs_wobs_c_desc.GetLength(I2);
    const auto C = depth2space_n_hobs_wobs_c_desc.GetLength(I3);
    assert(HoBs % / BlockSize == 0);
    assert(WoBs % / BlockSize == 0);
    const auto Ho = HoBs / BlockSize;
    const auto Wo = WoBs / BlockSize;

#define _depth2space_transform_ 5 // 3, 5 give the correct result

#if _depth2space_transform_ == 0
    const auto depth2space_n_ho_b0_wo_b1_c_desc = transform_tensor_descriptor(
        depth2space_n_hobs_wobs_c_desc,
        make_tuple(make_pass_through_transform(N),
                   make_unmerge_transform(make_tuple(Ho, BlockSize)),
                   make_unmerge_transform(make_tuple(Wo, BlockSize)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto conv_out_n_ho_wo_k_desc = transform_tensor_descriptor(
        depth2space_n_ho_b0_wo_b1_c_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(Ho),
                   make_pass_through_transform(Wo),
                   make_merge_transform(make_tuple(C, BlockSize, BlockSize))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<5, 2, 4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#elif _depth2space_transform_ == 1
    const auto depth2space_n_b0_ho_b1_wo_c_desc = transform_tensor_descriptor(
        depth2space_n_hobs_wobs_c_desc,
        make_tuple(make_pass_through_transform(N),
                   make_unmerge_transform(make_tuple(BlockSize, Ho)),
                   make_unmerge_transform(make_tuple(BlockSize, Wo)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto conv_out_n_ho_wo_k_desc = transform_tensor_descriptor(
        depth2space_n_b0_ho_b1_wo_c_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(Ho),
                   make_pass_through_transform(Wo),
                   make_merge_transform(make_tuple(C, BlockSize, BlockSize))),
        make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<4>{}, Sequence<5, 1, 3>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#elif _depth2space_transform_ == 2
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
                   make_merge_transform(make_tuple(C, BlockSize, BlockSize))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<5, 3, 4>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#elif _depth2space_transform_ == 3
    const auto depth2space_n_ho_b0_wo_b1_c_desc = transform_tensor_descriptor(
        depth2space_n_hobs_wobs_c_desc,
        make_tuple(make_pass_through_transform(N),
                   make_unmerge_transform(make_tuple(Ho, BlockSize)),
                   make_unmerge_transform(make_tuple(Wo, BlockSize)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto conv_out_n_ho_wo_k_desc = transform_tensor_descriptor(
        depth2space_n_ho_b0_wo_b1_c_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(Ho),
                   make_pass_through_transform(Wo),
                   make_merge_transform(make_tuple(BlockSize, BlockSize, C))),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<3>{}, Sequence<2, 4, 5>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#elif _depth2space_transform_ == 4
    const auto depth2space_n_b0_ho_b1_wo_c_desc = transform_tensor_descriptor(
        depth2space_n_hobs_wobs_c_desc,
        make_tuple(make_pass_through_transform(N),
                   make_unmerge_transform(make_tuple(BlockSize, Ho)),
                   make_unmerge_transform(make_tuple(BlockSize, Wo)),
                   make_pass_through_transform(C)),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
        make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

    const auto conv_out_n_ho_wo_k_desc = transform_tensor_descriptor(
        depth2space_n_b0_ho_b1_wo_c_desc,
        make_tuple(make_pass_through_transform(N),
                   make_pass_through_transform(Ho),
                   make_pass_through_transform(Wo),
                   make_merge_transform(make_tuple(BlockSize, BlockSize, C))),
        make_tuple(Sequence<0>{}, Sequence<2>{}, Sequence<4>{}, Sequence<1, 3, 5>{}),
        make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
#elif _depth2space_transform_ == 5
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
#endif
    return conv_out_n_ho_wo_k_desc;
#undef _depth2space_transform_
}

} // namespace ck
#endif
