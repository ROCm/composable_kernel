#include <string_view>
#include <tuple>
#include <array>
#include <utility>
#include <type_traits>
#include <cstring>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

template <typename T>
void reference_im2col(Tensor<T>& in_mtx_host_ref,
                      const Tensor<T>& in_host,
                      int /*N*/,
                      int /*K*/,
                      int C,
                      int /*Y*/,
                      int X,
                      int Hi,
                      int Wi,
                      int Ho,
                      int Wo,
                      int ConvStrideH,
                      int ConvStrideW,
                      int ConvDilationH,
                      int ConvDilationW,
                      int InLeftPadH,
                      int InLeftPadW,
                      int /*InRightPadH*/,
                      int /*InRightPadW*/)
{
    int GemmM = in_mtx_host_ref.GetLengths()[0];
    int GemmK = in_mtx_host_ref.GetLengths()[1];

    for(int gemm_m = 0; gemm_m < GemmM; ++gemm_m)
    {
        int mtmp = gemm_m;
        int n    = mtmp / (Ho * Wo);
        mtmp -= n * Ho * Wo;
        int ho = mtmp / Wo;
        int wo = mtmp - ho * Wo;

        for(int gemm_k = 0; gemm_k < GemmK; ++gemm_k)
        {
            int ktmp = gemm_k;
            int y    = ktmp / (X * C);
            ktmp -= y * X * C;
            int x = ktmp / C;
            int c = ktmp - x * C;

            int hi = y * ConvDilationH + ho * ConvStrideH - InLeftPadH;
            int wi = x * ConvDilationW + wo * ConvStrideW - InLeftPadW;

            bool inbound = (hi >= 0 && hi < Hi && wi >= 0 && wi < Wi);

            in_mtx_host_ref(gemm_m, gemm_k) = inbound ? in_host(n, hi, wi, c) : 0;
        }
    }
}

template <ck::index_t NDimSpatial,
          typename T,
          ck::index_t kBlockSize,
          ck::index_t kMPerBlock,
          ck::index_t kKPerBlock>
struct Im2Col
{
    __host__ __device__ static constexpr auto MakeBlockCopyTileDistribution()
    {
        using namespace ck;
        using namespace ck::tile_program;

        constexpr index_t NumWarp = kBlockSize / get_warp_size();

        constexpr index_t K1 = 16 / sizeof(T);
        constexpr index_t K0 = kKPerBlock / K1;

        constexpr index_t M2 = get_warp_size() / K0;
        constexpr index_t M1 = NumWarp;
        constexpr index_t M0 = kMPerBlock / (M1 * M2);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    __host__ __device__ void
    operator()(const std::array<ck::index_t, NDimSpatial + 2>& a_n_wis_c_lengths,
               const std::array<ck::index_t, NDimSpatial + 2>& /* a_n_wis_c_strides */,
               const std::array<ck::index_t, NDimSpatial + 2>& b_k_xs_c_lengths,
               const std::array<ck::index_t, NDimSpatial + 2>& /* b_k_xs_c_strides */,
               const std::array<ck::index_t, NDimSpatial + 2>& c_n_wos_k_lengths,
               const std::array<ck::index_t, NDimSpatial + 2>& /* c_n_wos_k_strides */,
               const std::array<ck::index_t, NDimSpatial>& conv_filter_strides,
               const std::array<ck::index_t, NDimSpatial>& conv_filter_dilations,
               const std::array<ck::index_t, NDimSpatial>& input_left_pads,
               const std::array<ck::index_t, NDimSpatial>& input_right_pads,
               const std::array<ck::index_t, 2> a_gemmm_gemmk_lengths,
               const std::array<ck::index_t, 2> a_gemmm_gemmk_strides,
               const T* p_a_img,
               T* p_a_mtx)
    {
        using namespace ck;
        using namespace ck::tile_program;

        const index_t N = a_n_wis_c_lengths[0];
        const index_t C = a_n_wis_c_lengths[3];

        const index_t Hi = a_n_wis_c_lengths[1];
        const index_t Wi = a_n_wis_c_lengths[2];

        const index_t Ho = c_n_wos_k_lengths[1];
        const index_t Wo = c_n_wos_k_lengths[2];

        const index_t Y = b_k_xs_c_lengths[1];
        const index_t X = b_k_xs_c_lengths[2];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const auto a_n_hi_wi_c = make_naive_tensor_view_packed<AddressSpaceEnum::Global>(
            p_a_img, make_tuple(N, Hi, Wi, C), Number<32>{});

        const auto a_n_hip_wip_c = transform_tensor_view(
            a_n_hi_wi_c,
            make_tuple(make_pass_through_transform(N),
                       make_pad_transform(Hi, InLeftPadH, InRightPadH),
                       make_pad_transform(Wi, InLeftPadW, InRightPadW),
                       make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

        const auto a_n_y_ho_x_wo_c = transform_tensor_view(
            a_n_hip_wip_c,
            make_tuple(
                make_pass_through_transform(N),
                make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                make_pass_through_transform(C)),
            make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

        const auto src_gemmm_gemmk =
            transform_tensor_view(a_n_y_ho_x_wo_c,
                                  make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                             make_merge_transform(make_tuple(Y, X, C))),
                                  make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                                  make_tuple(Sequence<0>{}, Sequence<1>{}));

        auto dst_gemmm_gemmk = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a_mtx,
            make_tuple(a_gemmm_gemmk_lengths[0], a_gemmm_gemmk_lengths[1]),
            make_tuple(a_gemmm_gemmk_strides[0], a_gemmm_gemmk_strides[1]),
            Number<32>{},
            Number<1>{});

        const auto numGemmM = a_gemmm_gemmk_lengths[0];
        const auto numGemmK = a_gemmm_gemmk_lengths[1];

        const auto id_block = get_block_id();

        const auto num_tile_m = __builtin_amdgcn_readfirstlane(numGemmM / kMPerBlock);

        const auto block2tile = make_cluster_descriptor(make_tuple(num_tile_m));

        const auto i_gemmm_gemmk = block2tile.CalculateBottomIndex(make_multi_index(id_block));

        const auto iGemmM = __builtin_amdgcn_readfirstlane(i_gemmm_gemmk[0]) * kMPerBlock;

        // src window
        auto src_block_window =
            make_tile_window(src_gemmm_gemmk,
                             make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}),
                             {iGemmM, 0},
                             MakeBlockCopyTileDistribution());

        // dst window
        auto dst_block_window = make_tile_window(
            dst_gemmm_gemmk, make_tuple(Number<kMPerBlock>{}, Number<kKPerBlock>{}), {iGemmM, 0});

        index_t iGemmK = 0;

        do
        {
            const auto block_tile = load_tile(src_block_window);

            store_tile(dst_block_window, block_tile);

            move_tile_window(src_block_window, {0, kKPerBlock});
            move_tile_window(dst_block_window, {0, kKPerBlock});

            iGemmK += kKPerBlock;

        } while(iGemmK < numGemmK);
    }
};

int main()
{
    using DataType = ck::half_t;

    constexpr ck::index_t NumDimSpatial = 2;

    ck::index_t N  = 128;
    ck::index_t K  = 1;
    ck::index_t C  = 256;
    ck::index_t Y  = 3;
    ck::index_t X  = 3;
    ck::index_t Hi = 28;
    ck::index_t Wi = 28;
    ck::index_t Ho = 14;
    ck::index_t Wo = 14;

    std::array<ck::index_t, NumDimSpatial + 2> in_lengths{N, Hi, Wi, C};
    std::array<ck::index_t, NumDimSpatial + 2> in_strides{0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 2> wei_lengths{K, Y, X, C};
    std::array<ck::index_t, NumDimSpatial + 2> wei_strides{0, 0, 0, 1};

    std::array<ck::index_t, NumDimSpatial + 2> out_lengths{N, Ho, Wo, K};
    std::array<ck::index_t, NumDimSpatial + 2> out_strides{0, 0, 0, 1};

    std::partial_sum(rbegin(in_lengths),
                     std::prev(rend(in_lengths)),
                     std::next(rbegin(in_strides)),
                     std::multiplies<>{});
    std::partial_sum(rbegin(wei_lengths),
                     std::prev(rend(wei_lengths)),
                     std::next(rbegin(wei_strides)),
                     std::multiplies<>{});
    std::partial_sum(rbegin(out_lengths),
                     std::prev(rend(out_lengths)),
                     std::next(rbegin(out_strides)),
                     std::multiplies<>{});

    std::array<ck::index_t, NumDimSpatial> filter_strides{2, 2};
    std::array<ck::index_t, NumDimSpatial> filter_dilations{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_left_pads{1, 1};
    std::array<ck::index_t, NumDimSpatial> input_right_pads{1, 1};

    // matrix
    std::array<ck::index_t, 2> in_mtx_lengths{N * Ho * Wo, Y * X * C};
    std::array<ck::index_t, 2> in_mtx_strides{0, 1};

    std::partial_sum(rbegin(in_mtx_lengths),
                     std::prev(rend(in_mtx_lengths)),
                     std::next(rbegin(in_mtx_strides)),
                     std::multiplies<>{});

    // host verify
    Tensor<DataType> in_host(in_lengths, in_strides);
    Tensor<DataType> in_mtx_host_ref(in_mtx_lengths, in_mtx_strides);
    Tensor<DataType> in_mtx_host_dev(in_mtx_lengths, in_mtx_strides);

    std::cout << " image tensor element size: " << in_host.GetElementSize() << std::endl;
    std::cout << "matrix tensor element size: " << in_mtx_host_ref.GetElementSize() << std::endl;

    std::cout << " image tensor element space size: " << in_host.GetElementSpaceSize() << std::endl;
    std::cout << "matrix tensor element sapce size: " << in_mtx_host_ref.GetElementSpaceSize()
              << std::endl;

    ck::utils::FillUniformDistributionIntegerValue<DataType>{-5.f, 5.f}(in_host);

    reference_im2col(in_mtx_host_ref,
                     in_host,
                     N,
                     K,
                     C,
                     Y,
                     X,
                     Hi,
                     Wi,
                     Ho,
                     Wo,
                     filter_strides[0],
                     filter_strides[1],
                     filter_dilations[0],
                     filter_dilations[1],
                     input_left_pads[0],
                     input_left_pads[1],
                     input_right_pads[0],
                     input_right_pads[1]);

    DeviceMem in_buf(sizeof(DataType) * in_host.GetElementSpaceSize());
    DeviceMem in_mtx_buf(sizeof(DataType) * in_mtx_host_ref.GetElementSpaceSize());

    in_buf.ToDevice(in_host.mData.data());

    constexpr ck::index_t kBlockSize = 256;

    constexpr ck::index_t kGemmMPerBlock = 256;
    constexpr ck::index_t kGemmKPerBlock = 128;

    ck::index_t kGridSize = (N * Ho * Wo) / kGemmMPerBlock;

    float ave_time =
        launch_kernel(StreamConfig{nullptr, true},
                      Im2Col<2, DataType, kBlockSize, kGemmMPerBlock, kGemmKPerBlock>{},
                      kGridSize,
                      kBlockSize,
                      0,
                      in_lengths,
                      in_strides,
                      wei_lengths,
                      wei_strides,
                      out_lengths,
                      out_strides,
                      filter_strides,
                      filter_dilations,
                      input_left_pads,
                      input_right_pads,
                      in_mtx_lengths,
                      in_mtx_strides,
                      static_cast<DataType*>(in_buf.GetDeviceBuffer()),
                      static_cast<DataType*>(in_mtx_buf.GetDeviceBuffer()));

    std::size_t num_btype = sizeof(DataType) * in_host.GetElementSize() +
                            sizeof(DataType) * in_mtx_host_ref.GetElementSize();

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s" << std::endl;

    in_mtx_buf.FromDevice(in_mtx_host_dev.mData.data());

    return !ck::utils::check_err(in_mtx_host_dev, in_mtx_host_ref);
}
