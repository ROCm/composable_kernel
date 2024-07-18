// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// Macro function
// construct constexpr TensorAdaptor from constexpr encoding
// encoded_tensor_adaptor are Tuple of following objects:
//    1. encoded transforms (Array of fixed size). Each encoded transform is a Tuple of following:
//           1.1 name (IndexTransformEnum)
//           1.2 meta data for constructor of the transform
//           1.3 num of lower dimension (index_t)
//           1.4 lower dimension Ids (Array of fixed size)
//           1.5 num of up dimension (index_t)
//           1.6 upper dimension Ids (Array of fixed size)
//    2. num of transforms (index_t)
//    3. encoded bottom dimension Ids (Array of fixed size)
//    4. num of bottom dimension (index_t)
//    5. encoded top dimension Ids (Array of fixed size)
//    6. num of top dimension (index_t)
#define CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(encoded_tensor_adaptor)                            \
    [encoded_tensor_adaptor]() {                                                                  \
        using namespace ck;                                                                       \
                                                                                                  \
        constexpr auto encoded_transforms  = encoded_tensor_adaptor.template At<0>();             \
        constexpr index_t num_transform    = encoded_tensor_adaptor.template At<1>();             \
        constexpr auto encoded_bottom_dims = encoded_tensor_adaptor.template At<2>();             \
        constexpr index_t num_bottom_dim   = encoded_tensor_adaptor.template At<3>();             \
        constexpr auto encoded_top_dims    = encoded_tensor_adaptor.template At<4>();             \
        constexpr index_t num_top_dim      = encoded_tensor_adaptor.template At<5>();             \
                                                                                                  \
        constexpr auto trans = [&encoded_transforms, &num_transform]() {                          \
            return generate_tuple(                                                                \
                [&encoded_transforms](auto i) constexpr {                                         \
                    constexpr auto name        = encoded_transforms[i].template At<0>();          \
                    constexpr auto meta_data   = encoded_transforms[i].template At<1>();          \
                    constexpr auto num_low_dim = encoded_transforms[i].template At<2>();          \
                    constexpr auto num_up_dim  = encoded_transforms[i].template At<4>();          \
                                                                                                  \
                    STATIC_ASSERT(name == IndexTransformEnum::PassThrough ||                      \
                                      name == IndexTransformEnum::Pad ||                          \
                                      name == IndexTransformEnum::Embed ||                        \
                                      name == IndexTransformEnum::Merge ||                        \
                                      name == IndexTransformEnum::UnMerge,                        \
                                  "");                                                            \
                                                                                                  \
                    if constexpr(name == IndexTransformEnum::PassThrough)                         \
                    {                                                                             \
                        index_t pos  = 0;                                                         \
                        auto low_len = meta_data.template Pop<index_t>(pos);                      \
                                                                                                  \
                        return make_pass_through_transform(low_len);                              \
                    }                                                                             \
                    else if constexpr(name == IndexTransformEnum::Pad)                            \
                    {                                                                             \
                        index_t pos    = 0;                                                       \
                        auto low_len   = meta_data.template Pop<index_t>(pos);                    \
                        auto left_pad  = meta_data.template Pop<index_t>(pos);                    \
                        auto right_pad = meta_data.template Pop<index_t>(pos);                    \
                                                                                                  \
                        return make_pad_transform(low_len, left_pad, right_pad);                  \
                    }                                                                             \
                    else if constexpr(name == IndexTransformEnum::Embed)                          \
                    {                                                                             \
                        index_t pos  = 0;                                                         \
                        auto up_lens = meta_data.template Pop<Array<index_t, num_up_dim>>(pos);   \
                        auto coefficients =                                                       \
                            meta_data.template Pop<Array<index_t, num_up_dim>>(pos);              \
                                                                                                  \
                        return make_embed_transform(up_lens, coefficients);                       \
                    }                                                                             \
                    else if constexpr(name == IndexTransformEnum::Merge)                          \
                    {                                                                             \
                        index_t pos   = 0;                                                        \
                        auto low_lens = meta_data.template Pop<Array<index_t, num_low_dim>>(pos); \
                                                                                                  \
                        return make_merge_transform(low_lens);                                    \
                    }                                                                             \
                    else if constexpr(name == IndexTransformEnum::UnMerge)                        \
                    {                                                                             \
                        index_t pos  = 0;                                                         \
                        auto up_lens = meta_data.template Pop<Array<index_t, num_up_dim>>(pos);   \
                                                                                                  \
                        return make_unmerge_transform(up_lens);                                   \
                    }                                                                             \
                    else if constexpr(name == IndexTransformEnum::Replicate)                      \
                    {                                                                             \
                        index_t pos  = 0;                                                         \
                        auto up_lens = meta_data.template Pop<Array<index_t, num_up_dim>>(pos);   \
                                                                                                  \
                        return make_replicate_transform(up_lens);                                 \
                    }                                                                             \
                },                                                                                \
                Number<num_transform>{});                                                         \
        }();                                                                                      \
                                                                                                  \
        constexpr auto low_dim_idss = [&encoded_transforms, &num_transform]() {                   \
            return generate_tuple(                                                                \
                [&encoded_transforms](auto i) {                                                   \
                    constexpr auto num_low_dim = encoded_transforms[i].template At<2>();          \
                    constexpr auto low_dims    = encoded_transforms[i].template At<3>();          \
                                                                                                  \
                    return TO_SEQUENCE(low_dims, num_low_dim);                                    \
                },                                                                                \
                Number<num_transform>());                                                         \
        }();                                                                                      \
                                                                                                  \
        constexpr auto up_dim_idss = [&encoded_transforms, &num_transform] {                      \
            return generate_tuple(                                                                \
                [&encoded_transforms](auto i) {                                                   \
                    constexpr auto num_up_dim = encoded_transforms[i].template At<4>();           \
                    constexpr auto up_dims    = encoded_transforms[i].template At<5>();           \
                                                                                                  \
                    return TO_SEQUENCE(up_dims, num_up_dim);                                      \
                },                                                                                \
                Number<num_transform>());                                                         \
        }();                                                                                      \
                                                                                                  \
        constexpr auto bottom_dim_ids = TO_SEQUENCE(encoded_bottom_dims, num_bottom_dim);         \
        constexpr auto top_dim_ids    = TO_SEQUENCE(encoded_top_dims, num_top_dim);               \
                                                                                                  \
        return TensorAdaptor<remove_cvref_t<decltype(trans)>,                                     \
                             remove_cvref_t<decltype(low_dim_idss)>,                              \
                             remove_cvref_t<decltype(up_dim_idss)>,                               \
                             remove_cvref_t<decltype(bottom_dim_ids)>,                            \
                             remove_cvref_t<decltype(top_dim_ids)>>{trans};                       \
    }()

// Macro function
// construct static TensorAdaptor from constexpr encoding
// encoded_tensor_adaptor are Tuple of following objects:
//    1. encoded transforms (Array of fixed size). Each encoded transform is a Tuple of following:
//           1.1 name (IndexTransformEnum)
//           1.2 meta data for constructor of the transform
//           1.3 num of lower dimension (index_t)
//           1.4 lower dimension Ids (Array of fixed size)
//           1.5 num of up dimension (index_t)
//           1.6 upper dimension Ids (Array of fixed size)
//    2. num of transforms (index_t)
//    3. encoded bottom dimension Ids (Array of fixed size)
//    4. num of bottom dimension (index_t)
//    5. encoded top dimension Ids (Array of fixed size)
//    6. num of top dimension (index_t)
#define CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(encoded_tensor_adaptor)                      \
    [encoded_tensor_adaptor]() {                                                                   \
        using namespace ck;                                                                        \
                                                                                                   \
        constexpr auto encoded_transforms  = encoded_tensor_adaptor.template At<0>();              \
        constexpr index_t num_transform    = encoded_tensor_adaptor.template At<1>();              \
        constexpr auto encoded_bottom_dims = encoded_tensor_adaptor.template At<2>();              \
        constexpr index_t num_bottom_dim   = encoded_tensor_adaptor.template At<3>();              \
        constexpr auto encoded_top_dims    = encoded_tensor_adaptor.template At<4>();              \
        constexpr index_t num_top_dim      = encoded_tensor_adaptor.template At<5>();              \
                                                                                                   \
        constexpr auto trans = [&encoded_transforms, &num_transform]() {                           \
            return generate_tuple(                                                                 \
                [&encoded_transforms](auto i) constexpr {                                          \
                    constexpr auto name        = encoded_transforms[i].template At<0>();           \
                    constexpr auto meta_data   = encoded_transforms[i].template At<1>();           \
                    constexpr auto num_low_dim = encoded_transforms[i].template At<2>();           \
                    constexpr auto num_up_dim  = encoded_transforms[i].template At<4>();           \
                                                                                                   \
                    STATIC_ASSERT(name == IndexTransformEnum::PassThrough ||                       \
                                      name == IndexTransformEnum::Pad ||                           \
                                      name == IndexTransformEnum::Embed ||                         \
                                      name == IndexTransformEnum::Merge ||                         \
                                      name == IndexTransformEnum::UnMerge,                         \
                                  "");                                                             \
                                                                                                   \
                    if constexpr(name == IndexTransformEnum::PassThrough)                          \
                    {                                                                              \
                        constexpr index_t low_len = meta_data.template Get<index_t>(0);            \
                                                                                                   \
                        return make_pass_through_transform(Number<low_len>{});                     \
                    }                                                                              \
                    else if constexpr(name == IndexTransformEnum::Pad)                             \
                    {                                                                              \
                        constexpr index_t low_len = meta_data.template Get<index_t>(0);            \
                                                                                                   \
                        constexpr index_t left_pad =                                               \
                            meta_data.template Get<index_t>(sizeof(low_len));                      \
                                                                                                   \
                        constexpr index_t right_pad =                                              \
                            meta_data.template Pop<index_t>(sizeof(low_len) + sizeof(left_pad));   \
                                                                                                   \
                        return make_pad_transform(                                                 \
                            Number<low_len>{}, Number<left_pad>{}, Number<right_pad>{});           \
                    }                                                                              \
                    else if constexpr(name == IndexTransformEnum::Embed)                           \
                    {                                                                              \
                        constexpr auto up_lens =                                                   \
                            meta_data.template Get<Array<index_t, num_up_dim>>(0);                 \
                                                                                                   \
                        constexpr auto coefficients =                                              \
                            meta_data.template Get<Array<index_t, num_up_dim>>(sizeof(up_lens));   \
                                                                                                   \
                        return make_embed_transform(TO_TUPLE_OF_NUMBER(up_lens, num_up_dim),       \
                                                    TO_TUPLE_OF_NUMBER(coefficients, num_up_dim)); \
                    }                                                                              \
                    else if constexpr(name == IndexTransformEnum::Merge)                           \
                    {                                                                              \
                        constexpr auto low_lens =                                                  \
                            meta_data.template Get<Array<index_t, num_low_dim>>(0);                \
                                                                                                   \
                        return make_merge_transform(TO_TUPLE_OF_NUMBER(low_lens, num_low_dim));    \
                    }                                                                              \
                    else if constexpr(name == IndexTransformEnum::UnMerge)                         \
                    {                                                                              \
                        constexpr auto up_lens =                                                   \
                            meta_data.template Get<Array<index_t, num_up_dim>>(0);                 \
                                                                                                   \
                        return make_unmerge_transform(TO_TUPLE_OF_NUMBER(up_lens, num_up_dim));    \
                    }                                                                              \
                    else if constexpr(name == IndexTransformEnum::Replicate)                       \
                    {                                                                              \
                        constexpr auto up_lens =                                                   \
                            meta_data.template Get<Array<index_t, num_up_dim>>(0);                 \
                                                                                                   \
                        return make_replicate_transform(TO_TUPLE_OF_NUMBER(up_lens, num_up_dim));  \
                    }                                                                              \
                },                                                                                 \
                Number<num_transform>{});                                                          \
        }();                                                                                       \
                                                                                                   \
        constexpr auto low_dim_idss = [&encoded_transforms, &num_transform]() {                    \
            return generate_tuple(                                                                 \
                [&encoded_transforms](auto i) {                                                    \
                    constexpr auto num_low_dim = encoded_transforms[i].template At<2>();           \
                    constexpr auto low_dims    = encoded_transforms[i].template At<3>();           \
                                                                                                   \
                    return TO_SEQUENCE(low_dims, num_low_dim);                                     \
                },                                                                                 \
                Number<num_transform>());                                                          \
        }();                                                                                       \
                                                                                                   \
        constexpr auto up_dim_idss = [&encoded_transforms, &num_transform] {                       \
            return generate_tuple(                                                                 \
                [&encoded_transforms](auto i) {                                                    \
                    constexpr auto num_up_dim = encoded_transforms[i].template At<4>();            \
                    constexpr auto up_dims    = encoded_transforms[i].template At<5>();            \
                                                                                                   \
                    return TO_SEQUENCE(up_dims, num_up_dim);                                       \
                },                                                                                 \
                Number<num_transform>());                                                          \
        }();                                                                                       \
                                                                                                   \
        constexpr auto bottom_dim_ids = TO_SEQUENCE(encoded_bottom_dims, num_bottom_dim);          \
        constexpr auto top_dim_ids    = TO_SEQUENCE(encoded_top_dims, num_top_dim);                \
                                                                                                   \
        return TensorAdaptor<remove_cvref_t<decltype(trans)>,                                      \
                             remove_cvref_t<decltype(low_dim_idss)>,                               \
                             remove_cvref_t<decltype(up_dim_idss)>,                                \
                             remove_cvref_t<decltype(bottom_dim_ids)>,                             \
                             remove_cvref_t<decltype(top_dim_ids)>>{trans};                        \
    }()
