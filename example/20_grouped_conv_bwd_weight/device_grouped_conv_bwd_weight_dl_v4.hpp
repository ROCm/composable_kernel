// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

// #include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"

using InDataType  = F16;
using WeiDataType = F16;
using OutDataType = F16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;

template <typename X> struct Debug;

namespace ck {
template <typename T>
__device__ T warp_shuffle_down(const T& v_local, uint32_t lane_delta)
{
#if 0
    return  __shfl_down(v_local, lane_delta);
#elif 1
    static_assert(sizeof(T) == sizeof(int32_t), "wrong!");

    const int32_t v_remote_tmp = __builtin_amdgcn_ds_bpermute(
        (__lane_id() << 2) + (lane_delta << 2), bit_cast<int32_t>(v_local));

    return bit_cast<T>(v_remote_tmp);
#endif
}

template <typename GridwiseConvBwdWeight,
          index_t BlockSize,
          index_t MinimumOccupancy = 1>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
__launch_bounds__(BlockSize, MinimumOccupancy)
#endif
    kernel_grouped_conv_bwd_weight_dl_v4(typename GridwiseConvBwdWeight::Argument arg)
{
//#if(!defined(__HIP_DEVICE_COMPILE__))
    __shared__ char
        p_share_in[GridwiseConvBwdWeight::ShareMemInSize * GridwiseConvBwdWeight::NumTilePerBlock];
    __shared__ char
        p_share_out[GridwiseConvBwdWeight::ShareMemOutSize * GridwiseConvBwdWeight::NumTilePerBlock];
    
    GridwiseConvBwdWeight::template Run(arg, p_share_in, p_share_out);
//#else
//    ignore = arg;
//#endif
}

namespace tensor_operation {
namespace device {

template <index_t BlockSize,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename BlockTileSize, // input, without padding
          index_t FilterSize,     //
          typename FilterParam,   // tuple<dilation, stride, padding>
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          index_t NBatch,
          index_t NumWavePerTile,
          index_t InScalarPerVector,
          index_t OutScalarPerVector,
          index_t DstScalarPerVector,
          bool RequirePadding,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
struct GridwiseGroupedConv2DBwdWeightDlV4
{
    static constexpr index_t
    GetConvOut(index_t length, index_t filter, index_t dilation, index_t pad, index_t stride)
    {
        return (length + pad + pad - ((filter - 1) * dilation + 1)) / stride + 1;
    }
    template<index_t W, index_t ScalarPerVector>
    static constexpr index_t GetAlignedPackW()
    {
        constexpr index_t pakced_w = W / ScalarPerVector;
        if constexpr(pakced_w == 7)
        {
            return 8;
        }
        else
        {
            return WaveSize / (WaveSize / pakced_w);
        }
    }
    static constexpr index_t GetBatchPerWave() { return WaveSize / (FilterSize * FilterSize); }
    
    static constexpr index_t NDimSpatial = 2;
    static constexpr index_t NumVectorPerPixel = NBatch / DstScalarPerVector;
    static constexpr auto I0                 = Number<0>{};
    static constexpr auto I1                 = Number<1>{};

    static constexpr index_t WaveSize = 64;
    static constexpr index_t Tile_H   = BlockTileSize{}.At(I0);
    static constexpr index_t Tile_W   = BlockTileSize{}.At(I1);

    // Only support pad left = pad right for now
    static constexpr index_t Pad_H = tuple_element_t<2, FilterParam>{}.At(I0);
    static constexpr index_t Pad_W = tuple_element_t<2, FilterParam>{}.At(I1);

    static constexpr index_t Stride_H = tuple_element_t<1, FilterParam>{}.At(I0);
    static constexpr index_t Stride_W = tuple_element_t<1, FilterParam>{}.At(I1);

    // Only support dilation = 1 for now
    static constexpr index_t Dilation_Y = tuple_element_t<0, FilterParam>{}.At(I0);
    static constexpr index_t Dilation_X = tuple_element_t<0, FilterParam>{}.At(I1);

    static constexpr index_t Filter_Y = FilterSize;
    static constexpr index_t Filter_X = FilterSize;

    static constexpr index_t TileIn_H = Tile_H + 2 * Pad_H;
    static constexpr index_t TileIn_W = Tile_H + 2 * Pad_W;

    static constexpr index_t TileOut_H =
        GetConvOut(Tile_H, FilterSize, Dilation_Y, Pad_H, Stride_H);
    static constexpr index_t TileOut_W =
        GetConvOut(Tile_W, FilterSize, Dilation_X, Pad_W, Stride_W);

    static constexpr index_t TileIn_Align_W  = TileIn_W;
    static constexpr index_t TileOut_Align_W = TileOut_W;

    static constexpr index_t TileIn_Pack_W     = GetAlignedPackW<Tile_W, InScalarPerVector>();
    static constexpr index_t TileIn_Pack_Group = WaveSize / TileIn_Pack_W;
    static constexpr index_t TileIn_Pack_H     = math::integer_divide_ceil(Tile_H, TileIn_Pack_Group);
    static constexpr index_t TileIn_Align_H =
        math::max(TileIn_Pack_H * TileIn_Pack_Group + Pad_H, TileIn_H);

    static constexpr index_t TileOut_Pack_W     = GetAlignedPackW<TileOut_W, OutScalarPerVector>();
    static constexpr index_t TileOut_Pack_Group = WaveSize / TileIn_Pack_W;
    static constexpr index_t TileOut_Pack_H = math::integer_divide_ceil(TileOut_H, TileOut_Pack_Group);

    static constexpr index_t BatchPerWave      = GetBatchPerWave();
    static constexpr index_t BatchPerTile      = BatchPerWave * NumWavePerTile;
    static constexpr index_t TileOut_HPerBatch = math::integer_divide_ceil(TileOut_H, BatchPerTile);
    static constexpr index_t TileOut_Align_H =
        math::max(TileOut_Pack_H * TileOut_Pack_Group, TileOut_HPerBatch* BatchPerWave* NumWavePerTile);
    static constexpr index_t ShareMemInSize =
        TileIn_Align_H * TileIn_Align_W * sizeof(InDataType) * NBatch;
    static constexpr index_t ShareMemOutSize =
        TileOut_Align_H * TileOut_Align_W * sizeof(OutDataType) * NBatch;

    static constexpr index_t NumTilePerBlock = BlockSize / WaveSize / NumWavePerTile;

    using InDataVector  = typename vector_type<InDataType, DstScalarPerVector>::type;
    using OutDataVector = typename vector_type<OutDataType, DstScalarPerVector>::type;

    template <index_t TileH, index_t TileW, index_t ScalarPerVector, typename SrcType, typename DestVector>
    static void __device__ load_data_from_global(const SrcType* p,
                                          index_t lane_id,
                                          index_t n_stride,
                                          index_t h,
                                          index_t w,
                                          index_t h_stride,
                                          index_t w_stride,
                                          DestVector* p_scratch)
    {
        ignore = h;
        ignore = w;

        using SrcVector            = typename vector_type<SrcType, ScalarPerVector>::type;
        constexpr index_t PackW = TileW / ScalarPerVector;
        constexpr index_t AlignedPackW = GetAlignedPackW<TileW, ScalarPerVector>();
        static_assert(PackW < WaveSize);

        static_assert((AlignedPackW & (AlignedPackW - 1)) == 0, "aligned width is not power 2!");

        constexpr index_t NumGroup = WaveSize / AlignedPackW;
        constexpr index_t AlignedPackH    = math::integer_divide_ceil(TileH, NumGroup);
        constexpr index_t PackH = TileH / NumGroup;

        const index_t x      = lane_id % AlignedPackW;
        const index_t y_offset = lane_id / AlignedPackW;

        auto get_offset = [&](index_t y_, index_t packed_x_, index_t n_) {
            return (y_ * h_stride + packed_x_ * ScalarPerVector * w_stride + n_ * n_stride) / ScalarPerVector;
        };

        // todo: check with real width/height
        // and use OOB to avoid tynamic control flow.
        auto* p_base = reinterpret_cast<const SrcVector*>(p);
        if(x < PackW)
        {
            static_for<0, PackH, 1>{}([&](auto i) {
                const index_t y = y_offset + i * NumGroup;
                // load data
                SrcVector tmp[NBatch];
                static_for<0, NBatch, 1>{}([&](auto n) {
                    const index_t offset = get_offset(y, x, n);
                    tmp[n]               = p_base[offset];
                });

                // interleave data
                auto* p_scratch_base = p_scratch + i * NumVectorPerPixel * ScalarPerVector;
                if constexpr(DstScalarPerVector == 1)
                {
                    static_assert(NBatch == 1);
                    static_for<0, ScalarPerVector, 1>{}(
                        [&](auto j) { p_scratch_base[j * NumVectorPerPixel] = tmp[0][j.value]; });
                }
                else if constexpr(ScalarPerVector == 1)
                {
                    static_assert(DstScalarPerVector > 1);
                    static_for<0, NBatch, 1>{}([&](auto n) {
                        p_scratch_base[n / DstScalarPerVector][n % DstScalarPerVector] = tmp[n];
                    });
                }
                else
                {
                    static_for<0, ScalarPerVector, 1>{}([&](auto j) {
                        static_for<0, NBatch, 1>{}([&](auto n) {
                            p_scratch_base[j * NumVectorPerPixel + n / DstScalarPerVector]
                                          [n % DstScalarPerVector] = tmp[n][j.value];
                        });
                    });
                }
            });

            if constexpr(AlignedPackH != PackH)
            {
                if(y_offset < (TileH - NumGroup * PackH))
                {
                    constexpr auto i = PackH;
                    const index_t y  = y_offset + i * NumGroup;
                    // load data
                    SrcVector tmp[NBatch];
                    static_for<0, NBatch, 1>{}([&](auto n) {
                        const index_t offset = get_offset(y, x, n);
                        tmp[n]               = p_base[offset];
                    });

                    // interleave data
                    auto* p_scratch_base = p_scratch + i * NumVectorPerPixel * ScalarPerVector;
                    if constexpr(DstScalarPerVector == 1)
                    {
                        static_assert(NBatch == 1);
                        static_for<0, ScalarPerVector, 1>{}([&](auto j) {
                            p_scratch_base[j * NumVectorPerPixel] = tmp[0][j.value];
                        });
                    }
                    else if constexpr(ScalarPerVector == 1)
                    {
                        static_for<0, NBatch, 1>{}([&](auto n) {
                            p_scratch_base[n / DstScalarPerVector][n % DstScalarPerVector] = tmp[n];
                        });
                    }
                    else
                    {
                        static_for<0, ScalarPerVector, 1>{}([&](auto j) {
                            static_for<0, NBatch, 1>{}([&](auto n) {
                                p_scratch_base[j * NumVectorPerPixel + n / DstScalarPerVector]
                                              [n % DstScalarPerVector] = tmp[n][j.value];
                            });
                        });
                    }
                }
            }
        }
    }

    // todo handle pading in p_sharemem
    template <index_t TileH, index_t TileW, index_t TileW_Stride, index_t ScalarPerVector, typename DestVector>
    static void __device__ write_data_to_lds(index_t lane_id,
                                      const DestVector* p_scratch,
                                      DestVector* p_sharemem)
    {
        constexpr index_t PackW = TileW / ScalarPerVector;
        constexpr index_t AlignedPackW = GetAlignedPackW<TileW, ScalarPerVector>();
        static_assert(AlignedPackW < WaveSize);

        constexpr index_t NumGroup = WaveSize / AlignedPackW;
        constexpr index_t AlignedPackH    = math::integer_divide_ceil(TileH, NumGroup);
        constexpr index_t PackH = TileH / NumGroup;

        const index_t x          = lane_id % AlignedPackW;
        const index_t y_offset   = lane_id / AlignedPackW;

        auto get_offset = [&](index_t y_, index_t x_) {
            return y_ * TileW_Stride * NumVectorPerPixel + x_ * NumVectorPerPixel * ScalarPerVector;
        };

        if(x < PackW)
        {
            static_for<0, PackH, 1>{}([&](auto i) {
                const index_t y      = y_offset + i * NumGroup;
                const index_t offset = get_offset(y, x);
                static_for<0, NumVectorPerPixel * ScalarPerVector, 1>{}([&](auto j) {
                    p_sharemem[offset + j] = p_scratch[i * NumVectorPerPixel * ScalarPerVector + j];
                });
            });
            if constexpr(AlignedPackH != PackH)
            {
                //static_assert(NumWavePerTile == 1);
                if (y_offset < (TileH - NumGroup * PackH))
                {
                    constexpr auto i = PackH;
                    const index_t y      = y_offset + i * NumGroup;
                    const index_t offset = get_offset(y, x);
                    static_for<0, NumVectorPerPixel * ScalarPerVector, 1>{}([&](auto j) {
                        p_sharemem[offset + j] = p_scratch[i * NumVectorPerPixel * ScalarPerVector + j];
                    });           
                }
            }
        }
    }

    template <index_t TileH>
    static void __device__ run_conv_bwd_weight(index_t x,
                                        index_t y,
                                        index_t h,
                                        index_t w,
                                        index_t hout_base,
                                        InDataVector* p_share_in,
                                        OutDataVector* p_share_out,
                                        AccDataType& acc)
    {
        ignore      = h;
        ignore      = w;
        auto get_in = [&](index_t ho_, index_t wo_, index_t i_) {
            // p_share_in has been shift with left_pad
            index_t hi = (ho_ + hout_base) * Stride_H + y * Dilation_Y;
            index_t wi = wo_ * Stride_W + x * Dilation_X;
            return p_share_in[(hi * TileIn_Align_W + wi) * NumVectorPerPixel + i_];
        };
        auto get_out = [&](index_t ho_, index_t wo_, index_t i_) {
            return p_share_out[((ho_ + hout_base) * TileOut_Align_W + wo_) * NumVectorPerPixel + i_];
        };
        if(x < Filter_X && y < Filter_Y)
        {
            static_for<0, TileH, 1>{}([&](auto ho) {
              //for (index_t ho = 0; ho < TileH; ho++) {
                static_for<0, TileOut_W, 1>{}([&](auto wo) {
                // for (index_t wo = 0; wo < TileOut_W; wo ++) {
                    static_for<0, NumVectorPerPixel, 1>{}([&](auto i) {
                        auto v_in  = get_in(ho, wo, i);
                        auto v_out = get_out(ho, wo, i);
                        inner_product(v_in, v_out, acc);
                    });
                });
            });
        }
    }
    template <typename Argument>
    static void __device__ write_output(const Argument& arg, index_t g, index_t y, index_t x, WeiDataType acc)
    {
        const index_t Wei_G_Stride = arg.wei_g_k_c_xs_strides_[0];
        const index_t Y_Stride     = arg.wei_g_k_c_xs_strides_[3];
        const index_t X_Stride     = arg.wei_g_k_c_xs_strides_[4];

        if(y < Filter_Y && x < Filter_X)
        {
            auto p_wei = arg.p_wei_grid_ + Wei_G_Stride * g + y * Y_Stride + x * X_Stride;
            *p_wei     = acc;
        }
    }
    template <typename DstVector>
    static void __device__ dump_lds(DstVector* p, index_t totalcount, index_t length)
    {
        for (index_t i = 0; i < totalcount; i++)
        {
            if (i % length ==0)
            {
                printf("\n [%d]", i/length);
            }
            printf("%08x ", bit_cast<uint32_t>(p[i]));
        }
        printf("\n");
    }

    template <typename Argument>
    static void __device__ Run(Argument arg, char* p_share_in, char* p_share_out)
    {
        const index_t g_idx   = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const index_t wave_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WaveSize);
        const index_t lane_id = __lane_id();

        constexpr index_t ThreadPerBatch = WaveSize / BatchPerWave;
        //Debug<Sequence<TileIn_Pack_H, TileOut_Pack_H>> xx3;
        static_assert(Tile_H % NumWavePerTile == 0);
        static_assert(TileOut_H % NumWavePerTile == 0);
        InDataVector tmp_in[math::integer_divide_ceil(TileIn_Pack_H, NumWavePerTile) * NumVectorPerPixel * InScalarPerVector]    = {};
        OutDataVector tmp_out[math::integer_divide_ceil(TileOut_Pack_H, NumWavePerTile) * NumVectorPerPixel * OutScalarPerVector] = {};

        static_assert(NumTilePerBlock == 1 || NumWavePerTile == 1);

        static constexpr index_t spatial_offset = 3;
        const index_t n = arg.in_g_n_c_wis_lengths_[1];
        index_t num_loop = n / NumTilePerBlock / BatchPerWave - 1;
        index_t n_idx = n / NumTilePerBlock * wave_id;

        if(wave_id == NumTilePerBlock - 1)
        {
            n_idx    = n / NumTilePerBlock * (NumTilePerBlock - 1);
            num_loop = (n - n_idx) / BatchPerWave - 1;
        }

        // In
        const index_t hi = arg.in_g_n_c_wis_lengths_[spatial_offset + 0];
        const index_t wi = arg.in_g_n_c_wis_lengths_[spatial_offset + 1];

        const index_t hi_stride   = arg.in_g_n_c_wis_strides_[spatial_offset + 0];
        const index_t wi_stride   = arg.in_g_n_c_wis_strides_[spatial_offset + 1];
        const index_t in_g_stride = arg.in_g_n_c_wis_strides_[0];
        const index_t in_n_stride = arg.in_g_n_c_wis_strides_[1];

        // Out
        const index_t ho = arg.out_g_n_k_wos_lengths_[spatial_offset + 0];
        const index_t wo = arg.out_g_n_k_wos_lengths_[spatial_offset + 1];

        const index_t ho_stride    = arg.out_g_n_k_wos_strides_[spatial_offset + 0];
        const index_t wo_stride    = arg.out_g_n_k_wos_strides_[spatial_offset + 1];
        const index_t out_g_stride = arg.out_g_n_k_wos_strides_[0];
        const index_t out_n_stride = arg.out_g_n_k_wos_strides_[1];

        // Wei
        auto* p_in  = arg.p_in_grid_ + g_idx * in_g_stride + n_idx * in_n_stride;
        auto* p_out = arg.p_out_grid_ + g_idx * out_g_stride + n_idx * out_n_stride;

        constexpr index_t Copy_Tile_H = Tile_H / NumWavePerTile;
        constexpr index_t Copy_TileOut_H = TileOut_H / NumWavePerTile;
        if constexpr (NumWavePerTile > 1)
        {
            static_assert(RequirePadding == false);
            p_in += Copy_Tile_H * hi_stride * wave_id;
            p_out += Copy_TileOut_H * ho_stride * wave_id;
        }

        InDataVector* share_in = 
            reinterpret_cast<InDataVector*>(p_share_in);
        OutDataVector* share_out =
            reinterpret_cast<OutDataVector*>(p_share_out);
        if constexpr(NumTilePerBlock > 1)
        {
            share_in  = reinterpret_cast<InDataVector*>(p_share_in + ShareMemInSize * wave_id);
            share_out = reinterpret_cast<InDataVector*>(p_share_out + ShareMemOutSize * wave_id);
        }

        // init lds 0
        index_t cluster_id = threadIdx.x % (WaveSize * NumWavePerTile);
        auto init_pading   = [&](auto* share_vec, auto count) {
            static_for<0, math::integer_divide_ceil(count, WaveSize * NumWavePerTile), 1>{}([&](auto i) {
                if(cluster_id + i * WaveSize * NumWavePerTile < count)
                {
                    share_vec[cluster_id + i * WaveSize * NumWavePerTile] = {};
                }
            });
        };
        auto init_array_pading   = [&](auto* share_vec, auto element_count, auto array_count, index_t stride) {
            static_for<0, math::integer_divide_ceil(array_count, WaveSize * NumWavePerTile), 1>{}([&](auto i) {
                static_for<0, element_count, 1>{}([&](auto j) {
                    if(cluster_id + i * WaveSize * NumWavePerTile < array_count)
                    {
                        auto p = share_vec + (cluster_id + i * WaveSize * NumWavePerTile) * stride + j;
                         *p = {};
                    }
                });
            });
        };       
        constexpr index_t TopPadingSize = Pad_H * TileIn_Align_W * NumVectorPerPixel;
        constexpr index_t TileInEnd = (Tile_H + Pad_H) * TileIn_Align_W;
        constexpr index_t ButtomPaddingSize = (ShareMemInSize / (sizeof(InDataType) * NBatch) - TileInEnd) * NumVectorPerPixel;
        if constexpr(Pad_W > 0)
        {
            static_assert(ButtomPaddingSize >= 0);
            init_array_pading(share_in + TopPadingSize, Number<Pad_W * NumVectorPerPixel>{}, Number<Tile_H>{}, TileIn_Align_W * NumVectorPerPixel);
            init_array_pading(share_in + TopPadingSize + (Tile_W + Pad_W) * NumVectorPerPixel , Number<Pad_W * NumVectorPerPixel>{}, Number<Tile_H>{}, TileIn_Align_W * NumVectorPerPixel);
        }
        if constexpr(Pad_H > 0)
        {
            init_pading(share_in, Number<TopPadingSize>{});
            init_pading(share_in + TileInEnd * NumVectorPerPixel, Number<ButtomPaddingSize>{});
        }
        constexpr index_t TileOutEnd = TileOut_H * TileOut_Align_W;
        constexpr index_t OutButtomPaddingSize = (ShareMemOutSize / (sizeof(OutDataType) * NBatch) - TileOutEnd) * NumVectorPerPixel;
        init_pading(share_out + TileOutEnd * NumVectorPerPixel, Number<OutButtomPaddingSize>{});

        if constexpr (NumWavePerTile > 1)
        {
            block_sync_lds();
        }

        // prefetch 0
        load_data_from_global<Copy_Tile_H, Tile_W, InScalarPerVector>(
            p_in, lane_id, in_n_stride, hi, wi, hi_stride, wi_stride, tmp_in);
        load_data_from_global<Copy_TileOut_H, TileOut_W, OutScalarPerVector>(
            p_out, lane_id, out_n_stride, ho, wo, ho_stride, wo_stride, tmp_out);
        p_in += NBatch * in_n_stride;
        p_out += NBatch * out_n_stride;

        // adjust share memory offset
        if constexpr (NumWavePerTile > 1)
        {
            static_assert(RequirePadding == false);
            share_in += Copy_Tile_H * TileIn_Align_W * NumVectorPerPixel * wave_id;
            share_out += Copy_TileOut_H * TileOut_Align_W * NumVectorPerPixel * wave_id;
        }
        auto share_in_base = share_in;
        share_in += (TileIn_Align_W * Pad_H + Pad_W) * NumVectorPerPixel;

        write_data_to_lds<Copy_Tile_H, Tile_W, TileIn_Align_W, InScalarPerVector>(
            lane_id, tmp_in, share_in);
        write_data_to_lds<Copy_TileOut_H, TileOut_W, TileOut_Align_W, OutScalarPerVector>(
            lane_id, tmp_out, share_out);

        constexpr index_t TileOut_H_batch = math::integer_divide_ceil(Copy_TileOut_H, BatchPerWave);
        index_t hout_base                 = lane_id / ThreadPerBatch * TileOut_H_batch;
        if constexpr (NumWavePerTile > 1)
        {
            hout_base += Copy_TileOut_H * wave_id;
        }
        index_t x = (lane_id % ThreadPerBatch) % Filter_X;
        index_t y = (lane_id % ThreadPerBatch) / Filter_X;
        float acc = 0;

        //if (threadIdx.x == 0)
        //{
        //    dump_lds(reinterpret_cast<InDataVector*>(p_share_in), ShareMemInSize/sizeof(InDataVector), TileIn_Align_W);
        //   dump_lds(reinterpret_cast<OutDataVector*>(p_share_out), ShareMemOutSize/sizeof(OutDataVector), TileOut_Align_W);
        //}

        while(num_loop > 0)
        {

            load_data_from_global<Copy_Tile_H, Tile_W, InScalarPerVector>(
                p_in, lane_id, in_n_stride, hi, wi, hi_stride, wi_stride, tmp_in);
            load_data_from_global<Copy_TileOut_H, TileOut_W, OutScalarPerVector>(
                p_out, lane_id, out_n_stride, ho, wo, ho_stride, wo_stride, tmp_out);
            p_in += NBatch * in_n_stride;
            p_out += NBatch * out_n_stride;

            // do conv_bwd on 0
            run_conv_bwd_weight<TileOut_H_batch>(
                x, y, ho, wo, hout_base, share_in_base, share_out, acc);

            // write 0
            write_data_to_lds<Copy_Tile_H, Tile_W, TileIn_Align_W, InScalarPerVector>(
                lane_id, tmp_in, share_in);
            write_data_to_lds<Copy_TileOut_H, TileOut_W, TileOut_Align_W, OutScalarPerVector>(
                lane_id, tmp_out, share_out);
            num_loop--;
        };

        // tail
        {
            run_conv_bwd_weight<TileOut_H_batch>(x, y, ho, wo, hout_base, share_in_base, share_out, acc);
        }

        if constexpr(ThreadPerBatch == 32)
        {
            //static_assert(0);
            float acc_2 = warp_shuffle_down(acc, ThreadPerBatch);
            acc += acc_2;
        }
        else if constexpr(ThreadPerBatch == 9)
        {
            // todo optimization reduce operation.
            float acc_2 = warp_shuffle_down(acc, ThreadPerBatch);
            float acc_3 = warp_shuffle_down(acc, 2 * ThreadPerBatch);
            float acc_4 = warp_shuffle_down(acc, 3 * ThreadPerBatch);
            float acc_5 = warp_shuffle_down(acc, 4 * ThreadPerBatch);
            float acc_6 = warp_shuffle_down(acc, 5 * ThreadPerBatch);
            float acc_7 = warp_shuffle_down(acc, 6 * ThreadPerBatch);
            acc += acc_2 + acc_3 + acc_4 + acc_5 + acc_6 + acc_7;
        }
        if constexpr(NumTilePerBlock == 1 && NumWavePerTile == 1)
        {
            
            write_output(arg, g_idx, y, x, acc);
        }
        else
        {
            uint32_t* p_share_acc = reinterpret_cast<uint32_t*>(p_share_in);
            block_sync_lds();
            p_share_acc[threadIdx.x] = bit_cast<uint32_t>(acc);
            block_sync_lds();
            if(hout_base == 0 && wave_id == 0)
            {
                for(int i = 1; i < NumTilePerBlock * NumWavePerTile; i++)
                {
                    acc += bit_cast<float>(p_share_acc[i * WaveSize + lane_id]);
                }
                write_output(arg, g_idx, y, x, type_convert<WeiDataType>(acc));
            }
        }
    }
    struct Argument
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_strides)
            : p_in_grid_{p_in_grid},
              p_wei_grid_{p_wei_grid},
              p_out_grid_{p_out_grid},
              in_g_n_c_wis_lengths_(in_g_n_c_wis_lengths),
              in_g_n_c_wis_strides_(in_g_n_c_wis_strides),
              wei_g_k_c_xs_lengths_(wei_g_k_c_xs_lengths),
              wei_g_k_c_xs_strides_(wei_g_k_c_xs_strides),
              out_g_n_k_wos_lengths_(out_g_n_k_wos_lengths),
              out_g_n_k_wos_strides_(out_g_n_k_wos_strides)
        {
        }

        std::size_t GetWorkspaceSizeBytes() const { return 0; }

        const InDataType* p_in_grid_;
        WeiDataType* p_wei_grid_;
        const OutDataType* p_out_grid_;

        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_strides_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_strides_;
    };
};

template <index_t NDimSpatial,
          index_t BlockSize,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename BlockTileSize, // input, without include pading
          index_t FilterSize,    // seqence<x, y, [z]>
          typename FilterParam,   // tuple<dilation, stride, padding>
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          index_t NBatch,
          index_t NumWavePerTile,
          index_t InScalarPerVector,
          index_t OutScalarPerVector,
          index_t DstScalarPerVector,
          bool RequirePadding,
          typename ComputeTypeA = InDataType,
          typename ComputeTypeB = ComputeTypeA>
struct DeviceGroupedConvBwdWeightDlV4 : public DeviceGroupedConvBwdWeight<NDimSpatial,
                                                                          InLayout,
                                                                          WeiLayout,
                                                                          OutLayout,
                                                                          InDataType,
                                                                          WeiDataType,
                                                                          OutDataType,
                                                                          InElementwiseOperation,
                                                                          WeiElementwiseOperation,
                                                                          OutElementwiseOperation,
                                                                          ComputeTypeA,
                                                                          ComputeTypeB>
{
    using DeviceOp = DeviceGroupedConvBwdWeightDlV4;
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static_assert(NDimSpatial == 2);
    static_assert(FilterSize * FilterSize < 64);
    static_assert(RequirePadding == false);
    static_assert(NBatch % DstScalarPerVector == 0);
    static_assert(is_same_v<InElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<WeiElementwiseOperation, element_wise::PassThrough>);
    static_assert(is_same_v<OutElementwiseOperation, element_wise::PassThrough>);

    using GridwiseConvBwdWeight =
        GridwiseGroupedConv2DBwdWeightDlV4<BlockSize,
                                           InDataType,
                                           WeiDataType,
                                           OutDataType,
                                           BlockTileSize,
                                           FilterSize,
                                           FilterParam,
                                           InElementwiseOperation,
                                           WeiElementwiseOperation,
                                           OutElementwiseOperation,
                                           NBatch,
                                           NumWavePerTile,
                                           InScalarPerVector,
                                           OutScalarPerVector,
                                           DstScalarPerVector,
                                           RequirePadding>;

    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 ck::index_t split_k)
            : p_in_grid_{p_in_grid},
              p_wei_grid_{p_wei_grid},
              p_out_grid_{p_out_grid},
              out_element_op_{out_element_op},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              in_g_n_c_wis_lengths_(in_g_n_c_wis_lengths),
              in_g_n_c_wis_strides_(in_g_n_c_wis_strides),
              wei_g_k_c_xs_lengths_(wei_g_k_c_xs_lengths),
              wei_g_k_c_xs_strides_(wei_g_k_c_xs_strides),
              out_g_n_k_wos_lengths_(out_g_n_k_wos_lengths),
              out_g_n_k_wos_strides_(out_g_n_k_wos_strides),
              conv_filter_strides_(conv_filter_strides),
              conv_filter_dilations_(conv_filter_dilations),
              input_left_pads_(input_left_pads),
              input_right_pads_(input_right_pads),
              k_batch_{split_k}
        {
        }

        std::size_t GetWorkspaceSizeBytes() const { return 0; }

        const InDataType* p_in_grid_;
        WeiDataType* p_wei_grid_;
        const OutDataType* p_out_grid_;

        OutElementwiseOperation out_element_op_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;

        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> in_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> wei_g_k_c_xs_strides_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> out_g_n_k_wos_strides_;
        std::array<ck::index_t, NDimSpatial> conv_filter_strides_;
        std::array<ck::index_t, NDimSpatial> conv_filter_dilations_;
        std::array<ck::index_t, NDimSpatial> input_left_pads_;
        std::array<ck::index_t, NDimSpatial> input_right_pads_;
        const index_t k_batch_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        void ShowInfo(const Argument&) {}
        index_t CalculateGridSize(const Argument& arg) { return arg.in_g_n_c_wis_lengths_[0]; }

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            index_t gdx = CalculateGridSize(arg);

            float ave_time = 0;
            typename GridwiseConvBwdWeight::Argument conv_arg{arg.p_in_grid_,
                                                              arg.p_wei_grid_,
                                                              arg.p_out_grid_,
                                                              arg.in_g_n_c_wis_lengths_, 
                                                              arg.in_g_n_c_wis_strides_,
                                                              arg.wei_g_k_c_xs_lengths_,
                                                              arg.wei_g_k_c_xs_strides_,
                                                              arg.out_g_n_k_wos_lengths_,
                                                              arg.out_g_n_k_wos_strides_};

            constexpr index_t minimum_occupancy = 2;

            const auto kernel = kernel_grouped_conv_bwd_weight_dl_v4<GridwiseConvBwdWeight, BlockSize, minimum_occupancy>;

            ave_time += launch_and_time_kernel(
                stream_config, kernel, dim3(gdx), dim3(BlockSize), 0, conv_arg);

            return ave_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        constexpr index_t spatial_offset = 3;
        // In
        const index_t hi        = arg.in_g_n_c_wis_lengths_[spatial_offset + 0];
        const index_t wi        = arg.in_g_n_c_wis_lengths_[spatial_offset + 1];
        const index_t wi_stride = arg.in_g_n_c_wis_strides_[spatial_offset + 1];
        // Out
        //const index_t ho        = arg.out_g_n_k_wos_lengths_[spatial_offset + 0];
        const index_t wo        = arg.out_g_n_k_wos_lengths_[spatial_offset + 1];
        const index_t wo_stride = arg.out_g_n_k_wos_strides_[spatial_offset + 1];
        // Wei
        const index_t filter_y = arg.wei_g_k_c_xs_lengths_[spatial_offset + 0];
        const index_t filter_x = arg.wei_g_k_c_xs_lengths_[spatial_offset + 1];
        const index_t filter_k = arg.wei_g_k_c_xs_lengths_[1];
        const index_t filter_c = arg.wei_g_k_c_xs_lengths_[2];

        static constexpr index_t Tile_H     = BlockTileSize{}.At(I0);
        static constexpr index_t Tile_W     = BlockTileSize{}.At(I1);
        static constexpr index_t Pad_H      = tuple_element_t<2, FilterParam>{}.At(I0);
        static constexpr index_t Pad_W      = tuple_element_t<2, FilterParam>{}.At(I1);
        static constexpr index_t Stride_H   = tuple_element_t<1, FilterParam>{}.At(I0);
        static constexpr index_t Stride_W   = tuple_element_t<1, FilterParam>{}.At(I1);
        static constexpr index_t Dilation_Y = tuple_element_t<0, FilterParam>{}.At(I0);
        static constexpr index_t Dilation_X = tuple_element_t<0, FilterParam>{}.At(I1);

        if(filter_k != 1 || filter_c != 1)
        {
            return false;
        }
        if constexpr(RequirePadding == false)
        {
            if(hi != Tile_H || wi != Tile_W)
            {
                return false;
            }
        }
        if(filter_y != FilterSize || filter_x != FilterSize)
        {
            return false;
        }
        if(Pad_H != arg.input_left_pads_[0] || Pad_W != arg.input_left_pads_[1] ||
           Pad_H != arg.input_right_pads_[0] || Pad_W != arg.input_right_pads_[1])
        {
            return false;
        }
        if(Stride_H != arg.conv_filter_strides_[0] || Stride_W != arg.conv_filter_strides_[1])
        {
            return false;
        }
        if(Dilation_Y != arg.conv_filter_dilations_[0] || Dilation_X != arg.conv_filter_dilations_[1])
        {
            return false;
        }
        if(InScalarPerVector > 1)
        {
            if(wi % InScalarPerVector != 0)
            {
                return false;
            }
            if(wi_stride != 1)
            {
                return false;
            }
        }
        if(OutScalarPerVector > 1)
        {
            if(wo % OutScalarPerVector != 0)
            {
                return false;
            }
            if(wo_stride != 1)
            {
                return false;
            }
        }
        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const InDataType* p_in_grid,
                 WeiDataType* p_wei_grid,
                 const OutDataType* p_out_grid,
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                 const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                 const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                 const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op,
                 const ck::index_t split_k)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        b_g_n_c_wis_lengths, // input
                        b_g_n_c_wis_strides,
                        e_g_k_c_xs_lengths, // weight
                        e_g_k_c_xs_strides,
                        a_g_n_k_wos_lengths, // output
                        a_g_n_k_wos_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op,
                        split_k};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        void* p_wei_grid,
                        const void* p_out_grid,
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_lengths, // input
                        const std::array<index_t, NDimSpatial + 3>& b_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_lengths, // weight
                        const std::array<index_t, NDimSpatial + 3>& e_g_k_c_xs_strides,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<ck::index_t, NDimSpatial>& input_left_pads,
                        const std::array<ck::index_t, NDimSpatial>& input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op,
                        const ck::index_t split_k) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<WeiDataType*>(p_wei_grid),
                                          static_cast<const OutDataType*>(p_out_grid),
                                          b_g_n_c_wis_lengths, // input
                                          b_g_n_c_wis_strides,
                                          e_g_k_c_xs_lengths, // weight
                                          e_g_k_c_xs_strides,
                                          a_g_n_k_wos_lengths, // output
                                          a_g_n_k_wos_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op,
                                          split_k);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        index_t Pad_H = tuple_element_t<2, FilterParam>{}.At(I0);
        index_t Pad_W = tuple_element_t<2, FilterParam>{}.At(I1);

        index_t Stride_H = tuple_element_t<1, FilterParam>{}.At(I0);
        index_t Stride_W = tuple_element_t<1, FilterParam>{}.At(I1);

        index_t Dilation_H = tuple_element_t<1, FilterParam>{}.At(I0);
        index_t Dilation_W = tuple_element_t<1, FilterParam>{}.At(I1);

        // clang-format off
        str << "DeviceGroupedConvBwdWeightDlV4<"
            << NDimSpatial << ", "
            << BlockSize << ", "
            << InLayout::name << ", "
            << WeiLayout::name << ", "
            << OutLayout::name << ", "
            << "BlockTileSize<" << BlockTileSize{}.At(I0) << BlockTileSize{}.At(I1) << ">, "
            << "FilterSize<" << FilterSize << ","<< FilterSize << ">, "
            << "Dilation<" << Dilation_H << ", " << Dilation_W<< ">, "
            << "Stride<" << Stride_H << ", " << Stride_W<< ">, "
            << "Pad<" << Pad_H << ", " << Pad_W<< ">, "
            << "NBatch: " << NBatch<< ", "
            << "NumWavePerTile: " << NumWavePerTile<< ", "
            << "InScalarPerVector: " << InScalarPerVector<< ", "
            << "OutScalarPerVector: " << OutScalarPerVector<< ", "
            << "DstScalarPerVector: " << DstScalarPerVector<< ", "
            << "RequirePadding: " << RequirePadding << ">"
            << std::endl;
        // clang-format on

        return str.str();
    }

    size_t GetWorkSpaceSize(const BaseArgument* p_arg) const override
    {
        auto arg = dynamic_cast<const Argument*>(p_arg);
        if(arg)
        {
            return arg->GetWorkspaceSizeBytes();
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument structure!");
    }

    void SetWorkSpacePointer(BaseArgument* p_arg,
                             void* p_workspace,
                             const StreamConfig& = StreamConfig{}) const override
    {
        auto p_arg_ = dynamic_cast<Argument*>(p_arg);
        if(p_arg_)
        {
            p_arg_->p_workspace_ = p_workspace;
        }
        else
            throw std::runtime_error(
                "The argument pointer is not an object of "
                "DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle::Argument structure!");
    }
};
} // namesapce device
} // namespace tensor_operation
} // namespace ck
