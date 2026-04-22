// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/host/concat.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/common/utils.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

#include <type_traits>

namespace ck_tile {

template <typename AsDataType_,
          typename BsDataType_,
          typename DsDataType_,
          typename AccDataType_,
          typename ODataType_,
          typename DsLayout_,
          typename ELayout_,
          typename CDElementwise_,
          index_t kM_,
          index_t kN_,
          index_t MWave_,
          index_t NWave_,
          index_t MPerXdl_,
          index_t NPerXdl_,
          index_t KPerXdl_,
          bool isCTransposed_,
          bool FixedVectorSize_ = false,
          index_t VectorSizeC_  = 1>
struct PermuteNEpilogueProblem
{
    using AsDataType                       = remove_cvref_t<AsDataType_>;
    using BsDataType                       = remove_cvref_t<BsDataType_>;
    using AccDataType                      = remove_cvref_t<AccDataType_>;
    using ODataType                        = remove_cvref_t<ODataType_>;
    using DsDataType                       = remove_cvref_t<DsDataType_>;
    using DsLayout                         = remove_cvref_t<DsLayout_>;
    using ELayout                          = remove_cvref_t<ELayout_>;
    using CDElementwise                    = remove_cvref_t<CDElementwise_>;
    static constexpr index_t kBlockSize    = MWave_ * NWave_ * get_warp_size();
    static constexpr index_t kMPerBlock    = kM_;
    static constexpr index_t kNPerBlock    = kN_;
    static constexpr index_t MWave         = MWave_;
    static constexpr index_t NWave         = NWave_;
    static constexpr index_t MPerXdl       = MPerXdl_;
    static constexpr index_t NPerXdl       = NPerXdl_;
    static constexpr index_t KPerXdl       = KPerXdl_;
    static constexpr index_t isCTransposed = isCTransposed_;
    static constexpr bool FixedVectorSize  = FixedVectorSize_;
    static constexpr index_t VectorSizeC   = VectorSizeC_;
    static constexpr index_t NumDTensor    = DsDataType::size();

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");
};

template <typename Problem_, typename Policy_ = void>
struct PermuteNEpilogue
{
    using Problem     = remove_cvref_t<Problem_>;
    using AsDataType  = remove_cvref_t<typename Problem::AsDataType>;
    using BsDataType  = remove_cvref_t<typename Problem::BsDataType>;
    using AccDataType = remove_cvref_t<typename Problem::AccDataType>;
    using ODataType   = remove_cvref_t<typename Problem::ODataType>;
    using DsDataType  = remove_cvref_t<typename Problem::DsDataType>;
    using DsLayout    = remove_cvref_t<typename Problem::DsLayout>;

    static constexpr bool ADataTypeIsTuple = is_detected<is_tuple, AsDataType>::value;
    static constexpr bool BDataTypeIsTuple = is_detected<is_tuple, BsDataType>::value;

    using AsDataTypeTuple = std::conditional_t<ADataTypeIsTuple,
                                               remove_cvref_t<AsDataType>,
                                               remove_cvref_t<tuple<AsDataType>>>;

    using BsDataTypeTuple = std::conditional_t<BDataTypeIsTuple,
                                               remove_cvref_t<BsDataType>,
                                               remove_cvref_t<tuple<BsDataType>>>;

    using ADataType = remove_cvref_t<std::tuple_element_t<number<0>{}, AsDataTypeTuple>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<number<0>{}, BsDataTypeTuple>>;

    using ATypeToUse = std::conditional_t<std::is_same_v<ADataType, pk_int4_t> ||
                                              std::is_same_v<ADataType, pk_fp4_t>,
                                          BDataType,
                                          ADataType>;
    // Used for weight-only quantization kernel, B would be dequantized to the same data type as A
    using BTypeToUse = std::conditional_t<std::is_same_v<BDataType, pk_int4_t> ||
                                              std::is_same_v<BDataType, pk_fp4_t> ||
                                              sizeof(BDataType) < sizeof(ADataType),
                                          ADataType,
                                          BDataType>;

    using ELayout                          = remove_cvref_t<typename Problem::ELayout>;
    using CDElementwise                    = remove_cvref_t<typename Problem::CDElementwise>;
    static constexpr index_t kBlockSize    = Problem::kBlockSize;
    static constexpr index_t kMPerBlock    = Problem::kMPerBlock;
    static constexpr index_t kNPerBlock    = Problem::kNPerBlock;
    static constexpr index_t MWave         = Problem::MWave;
    static constexpr index_t NWave         = Problem::NWave;
    static constexpr index_t MPerXdl       = Problem::MPerXdl;
    static constexpr index_t NPerXdl       = Problem::NPerXdl;
    static constexpr index_t KPerXdl       = Problem::KPerXdl;
    static constexpr index_t isCTransposed = Problem::isCTransposed;
    static constexpr bool FixedVectorSize  = Problem::FixedVectorSize;
    static constexpr index_t VectorSizeC   = Problem::VectorSizeC;
    static constexpr index_t MPerIteration = MPerXdl * MWave;
    static constexpr index_t NPerIteration = NPerXdl * NWave;
    static constexpr index_t NumDTensor    = Problem::NumDTensor;
    static constexpr index_t MRepeat       = kMPerBlock / (MPerXdl * MWave);
    static constexpr index_t NRepeat       = kNPerBlock / (NPerXdl * NWave);

    CDElementwise elfunc_;

    // PermuteN epilogue does not support D tensors or non-passthrough elementwise operations.
    // If D tensor support is needed, use CShuffleEpilogue instead.
    static_assert(NumDTensor == 0,
                  "PermuteNEpilogue does not support D tensors. Use CShuffleEpilogue instead.");
    static_assert(std::is_same_v<CDElementwise, element_wise::PassThrough>,
                  "PermuteNEpilogue only supports PassThrough elementwise. "
                  "Use CShuffleEpilogue for custom elementwise operations.");

    CK_TILE_DEVICE PermuteNEpilogue(CDElementwise elfunc = CDElementwise{}) : elfunc_(elfunc) {};

    static_assert(NumDTensor == DsLayout::size(),
                  "The size of DsDataType and DsLayout should be the same");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "PermuteNEpilogue",
                      concat('x', MWave, NWave),
                      concat('x', MPerXdl, NPerXdl, KPerXdl),
                      VectorSizeC,
                      isCTransposed ? "CTransposed" : "CNotTransposed");
        // clang-format on
    }

    /**
     * @brief Get the vector store size for C tensor.
     *
     * @note The vector store size for output C tensor would depend on multiple factors
     *       like its data layout and warp gemm C transposition. In general it would
     *       be the number of consecutive elements in contiguous C dimension hold by
     *       single thread.
     *
     * @return The vector store size for C tensor.
     */
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeC()
    {
        if constexpr(FixedVectorSize)
        {
            return VectorSizeC;
        }
        constexpr index_t max_vector_size = 16;
        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(ODataType)));
        }
        else
        {
            static_assert(false, "Unsupported ELayout!");
        }
    }

    /**
     * @brief Get the vector store size for Di tensor.
     *
     * @return The vector store size for Di tensor.
     */
    template <index_t I>
    CK_TILE_HOST_DEVICE static constexpr index_t GetVectorSizeD(number<I> index)
    {
        constexpr index_t max_vector_size = 16;
        using DiDataType = remove_cvref_t<std::tuple_element_t<index.value, DsDataType>>;
        using DiLayout   = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
        if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
        {
            return std::min(static_cast<int>(NPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::ColumnMajor>)
        {
            return std::min(static_cast<int>(MPerIteration),
                            static_cast<int>(max_vector_size / sizeof(DiDataType)));
        }
        else
        {
            static_assert(false, "Unsupported DLayout!");
        }
        return max_vector_size / sizeof(DiDataType);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    using WG = WarpGemmDispatcher<ATypeToUse,
                                  BTypeToUse,
                                  AccDataType,
                                  MPerXdl,
                                  NPerXdl,
                                  KPerXdl,
                                  isCTransposed>;

    using CWarpDstr         = typename WG::CWarpDstr;
    using CWarpTensor       = typename WG::CWarpTensor;
    using CWarpDstrEncoding = typename WG::CWarpDstrEncoding;

    // TODO: Check if there would be nicer ways to overload rather than with EmptyScale or nullptr_t
    struct EmptyScale
    {
    };

    template <typename, typename = void>
    struct ScaleDataType
    {
        using DataType = float;
    };

    template <typename T>
    struct ScaleDataType<T, std::void_t<typename T::DataType>>
    {
        using DataType = typename T::DataType;
    };

    template <typename ODramWindow,
              typename OAccTile,
              typename DsDramWindows,
              typename ScaleM = EmptyScale,
              typename ScaleN = EmptyScale>
    CK_TILE_DEVICE auto operator()(ODramWindow& out_dram_window,
                                   const OAccTile& o_acc_tile,
                                   const DsDramWindows& ds_dram_windows,
                                   void* /* p_smem */,
                                   const ScaleM& scale_m = {},
                                   const ScaleN& scale_n = {})
    {
        static constexpr int RowsPerLane = CWarpTensor::get_thread_buffer_size();

        static_assert(MPerXdl % RowsPerLane == 0,
                      "PermuteN: MPerXdl must be divisible by per-lane row count.");
        constexpr int kM0 = MWave;
        constexpr int kM2 = RowsPerLane;
        constexpr int kM1 = MPerXdl / kM2;

        constexpr int kN0 = NWave;
        constexpr int kN1 = NPerXdl;
        constexpr int kN2 = NRepeat;

        using IntrThreadShuffleEncode =
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<kM0, kM1, kM2>, sequence<kN0, kN1, kN2>>,
                                       tuple<sequence<1, 2>, sequence<1, 2>>,
                                       tuple<sequence<0, 0>, sequence<1, 1>>,
                                       sequence<1, 2>,
                                       sequence<2, 2>>;
        constexpr auto dram_tile_distribution =
            make_static_tile_distribution(IntrThreadShuffleEncode{});

        auto d_dram_windows = generate_tuple(
            [&](auto idx) {
                return make_tile_window(ds_dram_windows[idx], dram_tile_distribution);
            },
            number<NumDTensor>{});

        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        auto shuffle_acc  = make_static_distributed_tensor<AccDataType>(dram_tile_distribution);
        auto c_out_tensor = make_static_distributed_tensor<ODataType>(dram_tile_distribution);

        // Optional scales (must share the same distribution to match per-thread indexing)
        constexpr bool has_scales =
            !std::is_same<ScaleM, EmptyScale>::value && !std::is_same<ScaleN, EmptyScale>::value;
        constexpr bool has_scalar_scales =
            std::is_same_v<ScaleM, AccDataType> && std::is_same_v<ScaleN, AccDataType>;

        // Tiles to hold row/col scales when present
        using SMType = typename ScaleDataType<ScaleM>::DataType;
        using SNType = typename ScaleDataType<ScaleN>::DataType;

        auto sm_tile = make_static_distributed_tensor<SMType>(dram_tile_distribution);
        auto sn_tile = make_static_distributed_tensor<SNType>(dram_tile_distribution);

        // Build windows only if non-scalar scales are provided
        auto scale_m_window = [&]() {
            if constexpr(has_scales && !has_scalar_scales)
            {
                return make_tile_window(scale_m, dram_tile_distribution);
            }
            else
            {
                return EmptyScale{};
            }
        }();
        auto scale_n_window = [&]() {
            if constexpr(has_scales && !has_scalar_scales)
            {
                return make_tile_window(scale_n, dram_tile_distribution);
            }
            else
            {
                return EmptyScale{};
            }
        }();

        static_for<0, MRepeat, 1>{}([&](auto mIter) {
            // Slice accumulators for this M repeat into the permuted layout
            shuffle_acc.get_thread_buffer() = o_acc_tile.get_y_sliced_thread_data(
                merge_sequences(sequence<mIter, 0>{}, c_warp_y_index_zeros),
                merge_sequences(sequence<1, NRepeat>{}, c_warp_y_lengths));

            // If non-scalar scales provided, load them with identical distribution
            if constexpr(has_scales && !has_scalar_scales)
            {
                sm_tile = load_tile(scale_m_window); // row scales in permuted layout
                sn_tile = load_tile(scale_n_window); // col scales in permuted layout
            }

            // Pack "rows per lane" with permuted N layout
            static_for<0, NRepeat, 1>{}([&](auto n_idx) {
                // source indices in shuffle_acc: (n_idx * product(Y) + row)
                const index_t plane = c_warp_y_lengths.product();

                // Fuse scale (if present) and convert
                static_for<0, kM2, 1>{}([&](auto m_lane) {
                    const int src = n_idx * plane + m_lane;   // source row in this N-plane
                    const int dst = n_idx + m_lane * NRepeat; // permuted N layout in output
                    AccDataType v = shuffle_acc.get_thread_buffer()[src];

                    if constexpr(has_scalar_scales)
                    {
                        v = static_cast<AccDataType>(v * scale_m * scale_n);
                    }
                    else if constexpr(has_scales && !has_scalar_scales)
                    {
                        const auto sm = static_cast<float>(sm_tile.get_thread_buffer()[dst]);
                        const auto sn = static_cast<float>(sn_tile.get_thread_buffer()[dst]);
                        v             = static_cast<AccDataType>(v * sm * sn);
                    }

                    c_out_tensor.get_thread_buffer()[dst] = type_convert<ODataType>(v);
                });
            });

            // store/update
            if constexpr(decltype(out_dram_window.get_bottom_tensor_view())::DstInMemOp ==
                         memory_operation_enum::set)
            {
                store_tile(out_dram_window, c_out_tensor);
            }
            else
            {
                update_tile(out_dram_window, c_out_tensor);
            }

            // advance output (and any D-tensors) by one MPerXdl*MWave chunk
            move_tile_window(out_dram_window, {number<MPerXdl * MWave>{}, number<0>{}});
            static_for<0, NumDTensor, 1>{}([&](auto idx) {
                move_tile_window(d_dram_windows[idx], {number<MPerXdl * MWave>{}, number<0>{}});
            });
        });
    }
};
} // namespace ck_tile
