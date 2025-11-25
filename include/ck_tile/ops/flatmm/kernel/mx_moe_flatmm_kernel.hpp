// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp"
#include "ck_tile/ops/flatmm/kernel/moe_flatmm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {

// MX MOE FlatMM Kernel - combines MX (FP4xFP4) with MOE routing
// Based on MXFlatmmKernel structure with MOE extensions from MoeFlatmmKernel
template <typename TilePartitioner_,
          typename MXFlatmmPipeline_,
          typename EpiloguePipeline_,
          MoeFlatmmKind kind,
          typename FusedActivation = moe::MoeSilu>
struct MXMoeFlatmmKernel
{
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using FlatmmPipeline  = remove_cvref_t<MXFlatmmPipeline_>;
    using BlockGemmShape =
        remove_cvref_t<typename MXFlatmmPipeline_::BlockGemmShape>;
    using EpiloguePipeline              = remove_cvref_t<EpiloguePipeline_>;
    using ALayout                       = remove_cvref_t<typename FlatmmPipeline::ALayout>;
    using BLayout                       = remove_cvref_t<typename FlatmmPipeline::BLayout>;
    using ELayout                       = remove_cvref_t<typename FlatmmPipeline::CLayout>;
    using DsLayout                      = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType                    = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t kBlockSize = FlatmmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = FlatmmPipeline::UsePersistentKernel;

    using ADataType = remove_cvref_t<typename FlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename FlatmmPipeline::BDataType>;
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using AccDataType  = float;
    using ActivationOp = FusedActivation;

    // MX-specific packing parameters (from MXFlatmmKernel)
    static constexpr int MThreadPerXdl = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr int NThreadPerXdl = BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr int KThreadPerXdl = 64 / MThreadPerXdl;

    static constexpr int APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr int BPackedSize = numeric_traits<BDataType>::PackedSize;

    static constexpr int MXdlPack = FlatmmPipeline::MXdlPack;
    static constexpr int NXdlPack = FlatmmPipeline::NXdlPack;
    static constexpr int KXdlPack = FlatmmPipeline::KXdlPack;

    static constexpr index_t NumDTensor = DsDataType::size();

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();
    static constexpr auto I4 = number<4>();
    static constexpr auto I5 = number<5>();

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    // MOE-specific parameters (from MoeFlatmmKernel)
    static constexpr bool IsInputGemm = kind != MoeFlatmmKind::kFFN_gemm2;
    static constexpr bool IsGateUp    = kind == MoeFlatmmKind::kFFN_gemm1_gate_up;

    static constexpr index_t kMPerBlock     = EpiloguePipeline::kMPerBlock;
    static constexpr index_t kNPerBlock     = EpiloguePipeline::kNPerBlock;
    static constexpr index_t MWave          = EpiloguePipeline::MWave;
    static constexpr index_t NWave          = EpiloguePipeline::NWave;
    static constexpr index_t MPerXdl        = EpiloguePipeline::MPerXdl;
    static constexpr index_t NPerXdl        = EpiloguePipeline::NPerXdl;
    static constexpr index_t KPerXdl        = EpiloguePipeline::KPerXdl;
    static constexpr index_t isCTransposed  = EpiloguePipeline::isCTransposed;
    static constexpr index_t kMPerIteration = MPerXdl * MWave;
    static constexpr index_t kNPerIteration = NPerXdl * NWave;
    static constexpr index_t kNRepeat       = kNPerBlock / kNPerIteration;

    static constexpr int OutputNPerBlock =
        IsGateUp ? TilePartitioner::NPerBlock / 2 : TilePartitioner::NPerBlock;

    // MX always uses FP4 for both A and B
    static constexpr bool MXFP4_Pipeline = true;
    static constexpr int MXFP4N_Pack     = 2;
    static constexpr int MXFP4K_Pack     = 2;
    static constexpr int N_Pack          = MXFP4N_Pack;
    static constexpr int K_Pack          = MXFP4K_Pack;

    // Kernel arguments structure
    template <class ScaleM     = FlatmmScalePointer<-1>,
              class ScaleN     = FlatmmScalePointer<-1>,
              class ExpertBias = FlatmmScalePointer<-1>>
    struct MXMoeFlatmmKernelArgs
    {
        const ck_tile::index_t* p_sorted_token_ids;
        const ck_tile::index_t* p_sorted_expert_ids;
        const ck_tile::index_t* p_max_token_id;
        const void* p_sorted_expert_weights;
        const void* a_ptr;
        const void* b_ptr;
        void* e_ptr;
        ck_tile::index_t NumTokens;
        ck_tile::index_t TopK;
        ck_tile::index_t M;
        ck_tile::index_t N;
        ck_tile::index_t K;
        ck_tile::index_t stride_A;
        ck_tile::index_t stride_B;
        ck_tile::index_t stride_C;
        ck_tile::index_t k_batch;
        ck_tile::index_t n_padded_zeros;
        ck_tile::index_t k_padded_zeros;
        ScaleM scale_m;
        ScaleN scale_n;
        ExpertBias exp_bias;
    };

    template <class ScaleM     = FlatmmScalePointer<-1>,
              class ScaleN     = FlatmmScalePointer<-1>,
              class ExpertBias = FlatmmScalePointer<-1>>
    CK_TILE_HOST static constexpr auto
    MakeKernelArgs(const MoeFlatmmHostArgs<ScaleM, ScaleN, ExpertBias>& hostArgs)
    {
        return MXMoeFlatmmKernelArgs<ScaleM, ScaleN, ExpertBias>{
            hostArgs.p_sorted_token_ids,
            hostArgs.p_sorted_expert_ids,
            hostArgs.p_max_token_id,
            hostArgs.p_sorted_expert_weights,
            hostArgs.a_ptr,
            hostArgs.b_ptr,
            hostArgs.e_ptr,
            hostArgs.NumTokens,
            hostArgs.TopK,
            hostArgs.M,
            hostArgs.N,
            hostArgs.K,
            hostArgs.stride_A,
            hostArgs.stride_B,
            hostArgs.stride_C,
            hostArgs.k_batch,
            hostArgs.n_padded_zeros,
            hostArgs.k_padded_zeros,
            hostArgs.scale_m,
            hostArgs.scale_n,
            hostArgs.exp_bias};
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat(
            '_', "mx_moe_flatmm", gemm_prec_str<ADataType, BDataType>, FlatmmPipeline::GetName());
    }

    static constexpr auto BlockSize() -> dim3 { return dim3(kBlockSize); }

    static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }

    template <class MXMoeFlatmmKernelArgs>
    static constexpr auto GridSize(const MXMoeFlatmmKernelArgs& kargs)
    {
        if constexpr(UsePersistentKernel)
        {
            hipDeviceProp_t prop;
            int deviceId = 0;

            constexpr int block_size = MXMoeFlatmmKernel::BlockSize().x;
            int dync_smem_size       = 0;
            int maxActiveBlocksPerCU = 0;

            [[maybe_unused]] auto e = hipGetDeviceProperties(&prop, deviceId);

            e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxActiveBlocksPerCU,
                reinterpret_cast<void*>(
                    kentry<1, MXMoeFlatmmKernel, MXMoeFlatmmKernelArgs>),
                block_size,
                dync_smem_size);

            const int persistent_block_size = prop.multiProcessorCount * maxActiveBlocksPerCU;
            const int total_work_tile_cnt   = TilePartitioner::GridSize(kargs.M, kargs.N);

            assert(kargs.k_batch == 1);
            return dim3(min(persistent_block_size, total_work_tile_cnt), 1, kargs.k_batch);
        }
        else
        {
            return dim3(TilePartitioner::GridSize(kargs.M, kargs.N), 1, kargs.k_batch);
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPingSize()
    {
        return max(FlatmmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }
    
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPongSize()
    {
        return FlatmmPipeline::GetSmemSize();
    }

    struct SplitKBatchOffset
    {
        template <class KernelArgs>
        __device__ SplitKBatchOffset(const KernelArgs& kargs, const std::size_t k_id = blockIdx.z)
        {
            const auto k_total = kargs.K;
            const auto k_split = integer_divide_ceil(k_total, kargs.k_batch);

            const std::size_t splitted_k_start = k_id * k_split;
            splitted_k = min(k_split, static_cast<index_t>(k_total - splitted_k_start));

            a_k_split_offset = splitted_k_start;
            b_k_split_offset = splitted_k_start;
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    // template <typename KernelArgs>
    // CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    // {
    //     if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
    //                  is_any_of<EDataType, fp16_t, bf16_t>::value)
    //     {
    //         return false;
    //     }

    //     if constexpr(UsePersistentKernel)
    //     {
    //         if(kargs.k_batch != 1)
    //         {
    //             return false;
    //         }
    //     }

    //     if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
    //     {
    //         if(kargs.stride_A < kargs.K || kargs.K % FlatmmPipeline::GetVectorSizeA() != 0)
    //         {
    //             return false;
    //         }
    //     }
    //     else
    //     {
    //         if(kargs.stride_A < kargs.M || kargs.M % FlatmmPipeline::GetVectorSizeA() != 0)
    //         {
    //             return false;
    //         }
    //     }

    //     if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
    //     {
    //         if(kargs.stride_B < kargs.N)
    //         {
    //             return false;
    //         }
    //     }
    //     else
    //     {
    //         if(kargs.stride_B < kargs.K)
    //         {
    //             return false;
    //         }
    //     }

    //     bool DTensorIsValid = true;

    //     if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
    //     {
    //         if(kargs.stride_C < kargs.N)
    //         {
    //             return false;
    //         }
    //     }
    //     else
    //     {
    //         if(kargs.stride_C < kargs.M)
    //         {
    //             return false;
    //         }
    //     }
    //     return DTensorIsValid;
    // }
 template <typename KernelArgs>
    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                     is_any_of<EDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                std::cerr << "Conditions not met for Kbatch >1 !" << std::endl;
                return false;
            }
        }
        if constexpr(UsePersistentKernel)
        {
            if(kargs.k_batch != 1)
            {
                std::cerr << "Persistent mode doesn't support Kbatch >1 !" << std::endl;
                return false;
            }
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % TilePartitioner::KPerBlock != 0 && FlatmmPipeline::kPadK == false)
            {
                std::cerr << "Can't support K that is not a multiple of KPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.K % FlatmmPipeline::GetVectorSizeA() != 0)
            {
                std::cerr << "K is not a multiple of vector load size for A tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
            {
                std::cerr << "Can't support M that is not a multiple of MPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.M % FlatmmPipeline::GetVectorSizeA() != 0)
            {
                std::cerr << "M is not a multiple of vector load size for A tensor!" << std::endl;
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            // if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
            // {
            //     std::cerr << "Can't support N that is not a multiple of NPerBlock"
            //                  " without padding!"
            //               << std::endl;
            //     return false;
            // }
            if(kargs.N % FlatmmPipeline::GetVectorSizeB() != 0)
            {
                std::cerr << "N is not a multiple of vector load size for B tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.K % TilePartitioner::KPerBlock != 0 && FlatmmPipeline::kPadK == false)
            {
                std::cerr << "Can't support K that is not a multiple of KPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.K % FlatmmPipeline::GetVectorSizeB() != 0)
            {
                std::cerr << "K is not a multiple of vector load size for B tensor!" << std::endl;
                return false;
            }
        }

        bool DTensorIsValid = {true};
        static_for<0, NumDTensor, 1>{}([&](auto index) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
            if(std::is_same_v<DiLayout, ELayout> == false)
            {
                DTensorIsValid = false;
            }
            if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
                {
                    CK_TILE_ERROR("Can't support N for tensor D that is not a multiple of "
                                  "NPerBlock without padding!");
                    DTensorIsValid = false;
                }
                if(kargs.N % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for D tensor!");
                    DTensorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
                {
                    CK_TILE_ERROR("Can't support M for tensor D that is not a multiple of "
                                  "MPerBlock without padding!");

                    DTensorIsValid = false;
                }
                if(kargs.M % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for D tensor!");
                    DTensorIsValid = false;
                }
            }
        });

        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.stride_C % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
            {
                std::cerr << "Can't support N that is not a multiple of NPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                std::cerr << "N is not a multiple of vector load size for C tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
            {
                std::cerr << "Can't support M that is not a multiple of MPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                std::cerr << "M is not a multiple of vector load size for C tensor!" << std::endl;
                return false;
            }
        }
        return DTensorIsValid;
    }

    // Tensor view creation with MOE expert routing
    template <memory_operation_enum DstInMemOp = IsInputGemm ? memory_operation_enum::set
                                                             : memory_operation_enum::atomic_add,
              typename KernelArgs>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const ADataType* a_ptr,
                        const BDataType* b_flat_ptr,
                        EDataType* e_ptr,
                        [[maybe_unused]] const AccDataType* exp_weight_ptr,
                        const int expert_id,
                        const KernelArgs& kargs,
                        const SplitKBatchOffset& splitk_batch_offset)
    {
        // A tensor view (token activations)
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(kargs.M, splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(splitk_batch_offset.splitted_k, kargs.M),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        // B tensor view (expert weights) - per-expert offset
        index_t kFlatK = kargs.K * BlockGemmShape::WarpTile::at(I1);
        index_t kFlatN = kargs.N * kargs.K / kFlatK;

        const auto& b_flat_tensor_view = [&]() {
            return make_naive_tensor_view<address_space_enum::global>(
                b_flat_ptr + expert_id * kFlatN * kFlatK,
                make_tuple(kFlatN - kargs.n_padded_zeros / NPerXdl, kFlatK),
                make_tuple(kFlatK, 1),
                number<FlatmmPipeline::GetVectorSizeB()>{},
                number<1>{});
        }();

        // C tensor view (output)
        const auto& c_tensor_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.M, kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(kargs.N, kargs.M),
                    make_tuple(kargs.stride_C, 1),
                    number<1>{},
                    number<1>{});
            }
        }();

        // MX scale tensors (from MXFlatmmKernel)
        auto scale_a = kargs.scale_m;
        auto scale_b = kargs.scale_n;

        static constexpr int BlockScaleSize = 32;
        const auto&& scale_packs_m = integer_divide_ceil(kargs.M, (MXdlPack * MThreadPerXdl));
        const auto&& scale_packs_n = integer_divide_ceil(kargs.N, (NXdlPack * NThreadPerXdl));
        const auto&& scale_packs_k = kargs.K / BlockScaleSize / (KXdlPack * KThreadPerXdl);

        // A scale tensor view
        const auto& scale_a_tensor_view = [&]() {
            const auto scale_a_naive_desc = make_naive_tensor_descriptor_packed(
                make_tuple(scale_packs_m, scale_packs_k, KThreadPerXdl, MThreadPerXdl));
            const auto scale_a_desc = transform_tensor_descriptor(
                scale_a_naive_desc,
                make_tuple(make_merge_transform(make_tuple(scale_packs_m, MThreadPerXdl)),
                           make_merge_transform(make_tuple(scale_packs_k, KThreadPerXdl))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tensor_view<address_space_enum::global>(
                reinterpret_cast<const int32_t*>(scale_a.ptr), scale_a_desc);
        }();

        // B scale tensor view - per-expert offset
        const auto& scale_b_tensor_view = [&]() {
            const auto scale_b_navie_desc = make_naive_tensor_descriptor_packed(
                make_tuple(scale_packs_n, scale_packs_k, KThreadPerXdl, NThreadPerXdl));
            const auto scale_b_desc = transform_tensor_descriptor(
                scale_b_navie_desc,
                make_tuple(make_merge_transform(make_tuple(scale_packs_n, NThreadPerXdl)),
                           make_merge_transform(make_tuple(scale_packs_k, KThreadPerXdl))),
                make_tuple(sequence<0, 3>{}, sequence<1, 2>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));

            return make_tensor_view<address_space_enum::global>(
                reinterpret_cast<const int32_t*>(scale_b.ptr) +
                    expert_id * scale_packs_n * scale_packs_k,
                scale_b_desc);
        }();

        return make_tuple(
            a_tensor_view, b_flat_tensor_view, c_tensor_view, scale_a_tensor_view, scale_b_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadM>{});
            }
        }();

        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view = views.at(I2);
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<OutputNPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(c_tensor_view,
                                       make_tuple(number<OutputNPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<FlatmmPipeline::kPadN, false>{});
            }
        }();

        return make_tuple(a_pad_view, views.at(I1), c_pad_view, views.at(I3), views.at(I4));
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto MakeGemmTileWindows(const PadView& views,
                                                   [[maybe_unused]] const index_t coord_m,
                                                   const index_t coord_n)
    {
        const auto& a_pad_view      = views.at(I0);
        const auto& b_flat_pad_view = views.at(I1);
        const auto& c_pad_view      = views.at(I2);

        const auto& a_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {0, 0});
            }
            else
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, 0});
            }
        }();

        constexpr bool isNonInterleaveGateUp = !IsGateUp || MXFP4_Pipeline;

        const auto& b_flat_block_window =
            make_tile_window(b_flat_pad_view,
                             make_tuple(number<FlatmmPipeline::flatNPerWarp>{},
                                        number<FlatmmPipeline::flatKPerWarp>{}),
                             {static_cast<int>(coord_n / BlockGemmShape::WarpTile::at(I1) /
                                               (isNonInterleaveGateUp ? 1 : 2)),
                              0});

        const int output_N_offset = IsGateUp ? coord_n / 2 : coord_n;

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<OutputNPerBlock>{}),
            {0, output_N_offset});

        static constexpr int BlockScaleSize = 32;

        auto scale_a_block_window = make_tile_window(
            views.at(I3),
            make_tuple(number<TilePartitioner::MPerBlock / MXdlPack>{},
                       number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
            {0, 0});

        auto scale_b_block_window = make_tile_window(
            views.at(I4),
            make_tuple(number<TilePartitioner::NPerBlock / NXdlPack>{},
                       number<TilePartitioner::KPerBlock / (BlockScaleSize * KXdlPack)>{}),
            {coord_n / NXdlPack, 0});

        return make_tuple(a_block_window,
                          b_flat_block_window,
                          c_block_window,
                          scale_a_block_window,
                          scale_b_block_window);
    }

    template <class MXMoeFlatmmKernelArgs>
    CK_TILE_DEVICE void operator()(MXMoeFlatmmKernelArgs kargs) const
    {
        auto tilePartitioner = TilePartitioner{kargs.M, kargs.N};
        const auto [iM, iN] = tilePartitioner.GetOutputTileIndex(blockIdx.x);
        const index_t coord_m      = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t coord_n      = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        this->operator()(kargs, coord_m, coord_n);
    }

    template <class MXMoeFlatmmKernelArgs>
    CK_TILE_DEVICE void operator()(MXMoeFlatmmKernelArgs kargs, index_t coord_m, index_t coord_n) const
    {
        // Similar structure to MoeFlatmmKernel::operator() but with MX pipeline
        const SplitKBatchOffset splitk_batch_offset(kargs);
        const ADataType* a_ptr = static_cast<const ADataType*>(kargs.a_ptr) +
                                 splitk_batch_offset.a_k_split_offset / APackedSize;
        const BDataType* b_flat_ptr = static_cast<const BDataType*>(kargs.b_ptr) +
                                      splitk_batch_offset.b_k_split_offset / BPackedSize;
        EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

        __shared__ char smem_ptr_ping[GetSmemPingSize()];
        __shared__ char smem_ptr_pong[GetSmemPongSize()];

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        // MOE routing metadata
        const auto* sorted_token_ids   = kargs.p_sorted_token_ids;
        const auto* sorted_expert_ids  = kargs.p_sorted_expert_ids;
        const auto* max_token_id       = kargs.p_max_token_id;
        const auto* sorted_exp_weights = static_cast<const AccDataType*>(kargs.p_sorted_expert_weights);

        // Full MOE routing and GEMM logic would go here
        // Following the pattern from moe_flatmm_kernel.hpp but using MX tensor views
        // This is a placeholder for the complete implementation
    }
};

} // namespace ck_tile
