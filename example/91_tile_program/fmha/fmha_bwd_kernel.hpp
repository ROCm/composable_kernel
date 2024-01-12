// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tile_program/tile/tile_window.hpp"

// S[seqlen_q, seqlen_k] = Q[seqlen_q, hdim_q] * K[seqlen_k, hdim_q]
// P[seqlen_q, seqlen_k] = Softmax(S[seqlen_q, seqlen_k])
// O[seqlen_q, hdim_v] = P[seqlen_q, seqlen_k] * V[hdim_v, seqlen_k]

template <typename TilePartitioner_, typename FmhaPipeline_, typename EpiloguePipeline_>
struct FmhaBwdKernel
{
    using TilePartitioner                   = ck::remove_cvref_t<TilePartitioner_>;
    using FmhaPipeline                      = ck::remove_cvref_t<FmhaPipeline_>;
    using EpiloguePipeline                  = ck::remove_cvref_t<EpiloguePipeline_>;
    static constexpr ck::index_t kBlockSize = FmhaPipeline::kBlockSize;

    using QDataType    = ck::remove_cvref_t<typename FmhaPipeline::QDataType>;
    using KDataType    = ck::remove_cvref_t<typename FmhaPipeline::KDataType>;
    using VDataType    = ck::remove_cvref_t<typename FmhaPipeline::VDataType>;
    using GemmDataType = ck::remove_cvref_t<typename FmhaPipeline::GemmDataType>;
    using LSEDataType  = ck::remove_cvref_t<typename FmhaPipeline::LSEDataType>;
    using AccDataType  = ck::remove_cvref_t<typename FmhaPipeline::AccDataType>;
    using DDataType    = ck::remove_cvref_t<typename FmhaPipeline::DDataType>;
    // using ZDataType           = ck::remove_cvref_t<typename FmhaPipeline::ZDataType>;
    using OGradDataType = ck::remove_cvref_t<typename FmhaPipeline::OGradDataType>;
    using QGradDataType = ck::remove_cvref_t<typename FmhaPipeline::QGradDataType>;
    using KGradDataType = ck::remove_cvref_t<typename FmhaPipeline::KGradDataType>;
    using VGradDataType = ck::remove_cvref_t<typename FmhaPipeline::VGradDataType>;

    struct Kargs
    {
        const void* q_ptr;
        const void* k_ptr;
        const void* v_ptr;
        const void* lse_ptr;
        const void* do_ptr;
        const void* d_ptr;
        // void* z_ptr;
        void* dq_ptr;
        void* dk_ptr;
        void* dv_ptr;

        ck::index_t seqlen_q;
        ck::index_t seqlen_k;
        ck::index_t hdim_q;
        ck::index_t hdim_v;

        float scale;

        ck::index_t stride_q;
        ck::index_t stride_k;
        ck::index_t stride_v;
        ck::index_t stride_o;
        // ck::index_t stride_dq;
        // ck::index_t stride_dk;
        // ck::index_t stride_dv;
        // ck::index_t stride_do;

        ck::index_t nhead_stride_q;
        ck::index_t nhead_stride_k;
        ck::index_t nhead_stride_v;
        ck::index_t nhead_stride_o;
        ck::index_t nhead_stride_lsed;
        // ck::index_t nhead_stride_dq;
        // ck::index_t nhead_stride_dk;
        // ck::index_t nhead_stride_dv;
        // ck::index_t nhead_stride_do;

        ck::index_t batch_stride_q;
        ck::index_t batch_stride_k;
        ck::index_t batch_stride_v;
        ck::index_t batch_stride_o;
        ck::index_t batch_stride_lsed;
        // ck::index_t batch_stride_dq;
        // ck::index_t batch_stride_dk;
        // ck::index_t batch_stride_dv;
        // ck::index_t batch_stride_do;
    };

    __host__ static constexpr Kargs MakeKargs(const void* q_ptr,
                                              const void* k_ptr,
                                              const void* v_ptr,
                                              const void* lse_ptr,
                                              const void* do_ptr,
                                              const void* d_ptr,
                                              // void* z_ptr,
                                              void* dq_ptr,
                                              void* dk_ptr,
                                              void* dv_ptr,
                                              ck::index_t seqlen_q,
                                              ck::index_t seqlen_k,
                                              ck::index_t hdim_q,
                                              ck::index_t hdim_v,
                                              float scale,
                                              ck::index_t stride_q,
                                              ck::index_t stride_k,
                                              ck::index_t stride_v,
                                              ck::index_t stride_o,
                                              ck::index_t nhead_stride_q,
                                              ck::index_t nhead_stride_k,
                                              ck::index_t nhead_stride_v,
                                              ck::index_t nhead_stride_o,
                                              ck::index_t nhead_stride_lsed,
                                              ck::index_t batch_stride_q,
                                              ck::index_t batch_stride_k,
                                              ck::index_t batch_stride_v,
                                              ck::index_t batch_stride_o,
                                              ck::index_t batch_stride_lsed)
    {
        return Kargs{q_ptr,
                     k_ptr,
                     v_ptr,
                     lse_ptr,
                     do_ptr,
                     d_ptr,
                     dq_ptr,
                     dk_ptr,
                     dv_ptr,
                     seqlen_q,
                     seqlen_k,
                     hdim_q,
                     hdim_v,
                     scale,
                     stride_q,
                     stride_k,
                     stride_v,
                     stride_o,
                     nhead_stride_q,
                     nhead_stride_k,
                     nhead_stride_v,
                     nhead_stride_o,
                     nhead_stride_lsed,
                     batch_stride_q,
                     batch_stride_k,
                     batch_stride_v,
                     batch_stride_o,
                     batch_stride_lsed};
    }

    __host__ static constexpr auto
    GridSize(ck::index_t batch_size_, ck::index_t nhead_, ck::index_t seqlen_k_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_k_);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        return ck::math::max(FmhaPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    __device__ void operator()(Kargs kargs) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_n, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_k);

        const index_t i_n0 = __builtin_amdgcn_readfirstlane(i_tile_n * FmhaPipeline::kN0);

        // for simplicity, batch stride we just modify the pointer
        const QDataType* q_ptr = reinterpret_cast<const QDataType*>(kargs.q_ptr) +
                                 i_nhead * kargs.nhead_stride_q + i_batch * kargs.batch_stride_q;
        const KDataType* k_ptr = reinterpret_cast<const KDataType*>(kargs.k_ptr) +
                                 i_nhead * kargs.nhead_stride_k + i_batch * kargs.batch_stride_k;
        const VDataType* v_ptr = reinterpret_cast<const VDataType*>(kargs.v_ptr) +
                                 i_nhead * kargs.nhead_stride_v + i_batch * kargs.batch_stride_v;
        const LSEDataType* lse_ptr = reinterpret_cast<const LSEDataType*>(kargs.lse_ptr) +
                                     i_nhead * kargs.nhead_stride_lsed +
                                     i_batch * kargs.batch_stride_lsed;
        const DDataType* d_ptr = reinterpret_cast<const DDataType*>(kargs.d_ptr) +
                                 i_nhead * kargs.nhead_stride_lsed +
                                 i_batch * kargs.batch_stride_lsed;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      i_nhead * kargs.nhead_stride_o +
                                      i_batch * kargs.batch_stride_o;
        QGradDataType* dq_ptr = reinterpret_cast<QGradDataType*>(kargs.dq_ptr) +
                                i_nhead * kargs.nhead_stride_q + i_batch * kargs.batch_stride_q;
        KGradDataType* dk_ptr = reinterpret_cast<KGradDataType*>(kargs.dk_ptr) +
                                i_nhead * kargs.nhead_stride_k + i_batch * kargs.batch_stride_k;
        VGradDataType* dv_ptr = reinterpret_cast<VGradDataType*>(kargs.dv_ptr) +
                                i_nhead * kargs.nhead_stride_v + i_batch * kargs.batch_stride_v;

        // Q/K/V/LSE/D/dO/dQ/dK/dV DRAM and DRAM window
        const auto q_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            q_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_q),
            make_tuple(kargs.stride_q, 1),
            Number<32>{},
            Number<1>{});

        const auto qt_dram =
            transform_tensor_view(q_dram,
                                  make_tuple(make_pass_through_transform(kargs.hdim_q),
                                             make_pass_through_transform(kargs.seqlen_q)),
                                  make_tuple(Sequence<1>{}, Sequence<0>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto k_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            k_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_q),
            make_tuple(kargs.stride_k, 1),
            Number<32>{},
            Number<1>{});

        const auto kt_dram =
            transform_tensor_view(k_dram,
                                  make_tuple(make_pass_through_transform(kargs.hdim_q),
                                             make_pass_through_transform(kargs.seqlen_k)),
                                  make_tuple(Sequence<1>{}, Sequence<0>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto v_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            v_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_v),
            make_tuple(kargs.stride_v, 1),
            Number<32>{},
            Number<1>{});

        auto lse_dram = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
            lse_ptr, make_tuple(kargs.seqlen_q), Number<32>{});

        auto d_dram = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
            d_ptr, make_tuple(kargs.seqlen_q), Number<32>{});

        auto do_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            do_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_o, 1),
            Number<32>{},
            Number<1>{});

        const auto dot_dram =
            transform_tensor_view(do_dram,
                                  make_tuple(make_pass_through_transform(kargs.hdim_v),
                                             make_pass_through_transform(kargs.seqlen_q)),
                                  make_tuple(Sequence<1>{}, Sequence<0>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));

        const auto dq_dram =
            make_naive_tensor_view<AddressSpaceEnum::Global, InMemoryDataOperationEnum::AtomicAdd>(
                dq_ptr,
                make_tuple(kargs.seqlen_q, kargs.hdim_q),
                make_tuple(kargs.stride_q, 1),
                Number<32>{},
                Number<1>{});

        auto q_dram_window = make_tile_window(
            q_dram,
            [&]() {
                if constexpr(FmhaPipeline::kQLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kM0>{},
                                      Number<FmhaPipeline::kQKHeaddim>{});
                else
                    return make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK0>{});
            }(),
            {0, 0});

        auto qt_dram_window =
            make_tile_window(qt_dram,
                             [&]() {
                                 if constexpr(FmhaPipeline::kQTLoadOnce)
                                     return make_tuple(Number<FmhaPipeline::kQKHeaddim>{},
                                                       Number<FmhaPipeline::kM0>{});
                                 else
                                     return make_tuple(Number<FmhaPipeline::kQKHeaddim>{},
                                                       Number<FmhaPipeline::kK3>{});
                             }(),
                             {0, 0});

        auto k_dram_window = make_tile_window(
            k_dram,
            [&]() {
                if constexpr(FmhaPipeline::kKLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kN0>{},
                                      Number<FmhaPipeline::kQKHeaddim>{});
                else
                    return make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK0>{});
            }(),
            {i_n0, 0});

        auto kt_dram_window =
            make_tile_window(kt_dram,
                             [&]() {
                                 if constexpr(FmhaPipeline::kKTLoadOnce)
                                     return make_tuple(Number<FmhaPipeline::kQKHeaddim>{},
                                                       Number<FmhaPipeline::kN0>{});
                                 else
                                     return make_tuple(Number<FmhaPipeline::kQKHeaddim>{},
                                                       Number<FmhaPipeline::kK4>{});
                             }(),
                             {0, i_n0});

        auto v_dram_window = make_tile_window(
            v_dram,
            [&]() {
                if constexpr(FmhaPipeline::kVLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kN0>{},
                                      Number<FmhaPipeline::kVHeaddim>{});
                else
                    return make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kK2>{});
            }(),
            {i_n0, 0});

        auto do_dram_window = make_tile_window(
            do_dram,
            [&]() {
                if constexpr(FmhaPipeline::kOGradLoadOnce)
                    return make_tuple(Number<FmhaPipeline::kM0>{},
                                      Number<FmhaPipeline::kVHeaddim>{});
                else
                    return make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kK2>{});
            }(),
            {0, 0});

        auto dot_dram_window =
            make_tile_window(dot_dram,
                             [&]() {
                                 if constexpr(FmhaPipeline::kOGradTLoadOnce)
                                     return make_tuple(Number<FmhaPipeline::kVHeaddim>{},
                                                       Number<FmhaPipeline::kM0>{});
                                 else
                                     return make_tuple(Number<FmhaPipeline::kVHeaddim>{},
                                                       Number<FmhaPipeline::kK1>{});
                             }(),
                             {0, 0});

        auto dq_dram_window = make_tile_window(
            dq_dram,
            make_tuple(Number<FmhaPipeline::kM0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
            {0, 0});

        auto lse_dram_window =
            make_tile_window(lse_dram, make_tuple(Number<FmhaPipeline::kM0>{}), {0});

        auto d_dram_window = make_tile_window(d_dram, make_tuple(Number<FmhaPipeline::kM0>{}), {0});

        auto [dk_acc_tile, dv_acc_tile] = FmhaPipeline{}(q_dram_window,
                                                         qt_dram_window,
                                                         k_dram_window,
                                                         kt_dram_window,
                                                         v_dram_window,
                                                         do_dram_window,
                                                         dot_dram_window,
                                                         lse_dram_window,
                                                         d_dram_window,
                                                         dq_dram_window,
                                                         kargs.scale,
                                                         kargs.seqlen_q / FmhaPipeline::kM0,
                                                         smem_ptr);

        const auto dk_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            dk_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_q),
            make_tuple(kargs.stride_k, 1),
            Number<32>{},
            Number<1>{});

        const auto dv_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            dv_ptr,
            make_tuple(kargs.seqlen_k, kargs.hdim_v),
            make_tuple(kargs.stride_v, 1),
            Number<32>{},
            Number<1>{});

        auto dk_dram_window = make_tile_window(
            dk_dram,
            make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kQKHeaddim>{}),
            {i_n0, 0});

        auto dv_dram_window = make_tile_window(
            dv_dram,
            make_tuple(Number<FmhaPipeline::kN0>{}, Number<FmhaPipeline::kVHeaddim>{}),
            {i_n0, 0});

        EpiloguePipeline{}(dk_dram_window, dv_dram_window, dk_acc_tile, dv_acc_tile);
    }
};

template <typename TilePartitioner_, typename FmhaOGradDotO_>
struct FmhaBwdOGradDotOKernel
{
    using TilePartitioner = ck::remove_cvref_t<TilePartitioner_>;
    using FmhaOGradDotO   = ck::remove_cvref_t<FmhaOGradDotO_>;

    using DDataType     = ck::remove_cvref_t<typename FmhaOGradDotO::DDataType>;
    using ODataType     = ck::remove_cvref_t<typename FmhaOGradDotO::ODataType>;
    using OGradDataType = ck::remove_cvref_t<typename FmhaOGradDotO::OGradDataType>;

    static constexpr ck::index_t kBlockSize = FmhaOGradDotO::kBlockSize;

    static constexpr ck::index_t kM0       = kBlockSize;
    static constexpr ck::index_t kVHeaddim = FmhaOGradDotO::kVHeaddim;

    struct Kargs
    {
        const void* o_ptr;
        const void* do_ptr;
        void* d_ptr;

        ck::index_t seqlen_q;
        ck::index_t hdim_v;

        ck::index_t stride_o;
        // ck::index_t stride_do;

        ck::index_t nhead_stride_o;
        ck::index_t nhead_stride_d;
        // ck::index_t nhead_stride_do;

        ck::index_t batch_stride_o;
        ck::index_t batch_stride_d;
        // ck::index_t batch_stride_do;
    };

    __host__ static constexpr Kargs MakeKargs(const void* o_ptr,
                                              const void* do_ptr,
                                              void* d_ptr,
                                              ck::index_t seqlen_q,
                                              ck::index_t hdim_v,
                                              ck::index_t stride_o,
                                              ck::index_t nhead_stride_o,
                                              ck::index_t nhead_stride_d,
                                              ck::index_t batch_stride_o,
                                              ck::index_t batch_stride_d)
    {
        return Kargs{o_ptr,
                     do_ptr,
                     d_ptr,
                     seqlen_q,
                     hdim_v,
                     stride_o,
                     nhead_stride_o,
                     nhead_stride_d,
                     batch_stride_o,
                     batch_stride_d};
    }

    __host__ static constexpr auto
    GridSize(ck::index_t batch_size_, ck::index_t nhead_, ck::index_t seqlen_q_)
    {
        return TilePartitioner::GridSize(batch_size_, nhead_, seqlen_q_);
    }

    __host__ static constexpr auto BlockSize() { return dim3(kBlockSize); }

    __host__ __device__ static constexpr ck::index_t GetSmemSize() { return 0; }

    __device__ void operator()(Kargs kargs) const
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // allocate LDS
        // __shared__ char smem_ptr[GetSmemSize()];

        // divide problem
        const auto [i_tile_m, i_nhead, i_batch] = TilePartitioner{}(kargs.seqlen_q);

        const index_t i_m0 = __builtin_amdgcn_readfirstlane(i_tile_m * kM0);

        // for simplicity, batch stride we just modify the pointer
        const ODataType* o_ptr = reinterpret_cast<const ODataType*>(kargs.o_ptr) +
                                 i_nhead * kargs.nhead_stride_o + i_batch * kargs.batch_stride_o;
        const OGradDataType* do_ptr = reinterpret_cast<const OGradDataType*>(kargs.do_ptr) +
                                      i_nhead * kargs.nhead_stride_o +
                                      i_batch * kargs.batch_stride_o;
        DDataType* d_ptr = reinterpret_cast<DDataType*>(kargs.d_ptr) +
                           i_nhead * kargs.nhead_stride_d + i_batch * kargs.batch_stride_d;

        // O/dO/D DRAM and DRAM window
        auto o_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            o_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_o, 1),
            Number<32>{},
            Number<1>{});

        auto do_dram = make_naive_tensor_view<AddressSpaceEnum::Global>(
            do_ptr,
            make_tuple(kargs.seqlen_q, kargs.hdim_v),
            make_tuple(kargs.stride_o, 1),
            Number<32>{},
            Number<1>{});

        auto d_dram = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
            d_ptr, make_tuple(kargs.seqlen_q), Number<32>{});

        auto o_dram_window =
            make_tile_window(o_dram, make_tuple(Number<kM0>{}, Number<kVHeaddim>{}), {i_m0, 0});

        auto do_dram_window =
            make_tile_window(do_dram, make_tuple(Number<kM0>{}, Number<kVHeaddim>{}), {i_m0, 0});

        auto d_dram_window = make_tile_window(d_dram, make_tuple(Number<kM0>{}), {i_m0});

        FmhaOGradDotO{}(o_dram_window, do_dram_window, d_dram_window);
    }
};
