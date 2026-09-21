// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/tdm.hpp"
#include "ck_tile/host/kernel_launch.hpp"

namespace ck_tile {
namespace test {

using F16 = half_t;
using F8  = fp8_t;

using Row = tensor_layout::gemm::RowMajor;
using Col = tensor_layout::gemm::ColumnMajor;

using GatherModeEnable  = bool_constant<true>;
using GatherModeDisable = bool_constant<false>;

using Gather16bitIndex = constant<TDMGatherIndexSize::Row16bit_Index>;
using Gather32bitIndex = constant<TDMGatherIndexSize::Row32bit_Index>;

struct TDMTestParams
{
    index_t m         = 16;
    index_t n         = 16;
    index_t x_stride  = -1;
    index_t y_stride  = -1;
    int do_validation = 1;
    int warmup        = 0;
    int repeat        = 1;

    template <typename Layout>
    void normalize()
    {
        if constexpr(std::is_same_v<Layout, tensor_layout::gemm::RowMajor>)
        {
            if(x_stride < 0)
                x_stride = n;
            if(y_stride < 0)
                y_stride = n;
        }
        else
        {
            if(x_stride < 0)
                x_stride = m;
            if(y_stride < 0)
                y_stride = m;
        }
    }
};

using TestTypes = ::testing::Types<std::tuple<F16, Row>,
                                   std::tuple<F16, Col>,
                                   std::tuple<F16, Row, Gather16bitIndex>,
                                   std::tuple<F16, Col, Gather16bitIndex>,
                                   std::tuple<F8, Row>,
                                   std::tuple<F8, Col>,
                                   std::tuple<F8, Row, Gather16bitIndex>,
                                   std::tuple<F8, Col, Gather16bitIndex>>;

template <typename TypeParam>
class TDMBasicTypedTest : public ::testing::Test
{
    protected:
    using DataType   = std::tuple_element_t<0, TypeParam>;
    using Layout     = std::tuple_element_t<1, TypeParam>;
    using GatherMode = std::
        conditional_t<std::tuple_size<TypeParam>::value == 3, GatherModeEnable, GatherModeDisable>;

    template <typename T, bool Enable>
    struct GatherModeDTypeHelper
    {
        using type = uint16_t; // dummy data type when gather mode is disabled
    };

    template <typename T>
    struct GatherModeDTypeHelper<T, true>
    {
        using type =
            std::conditional_t<std::tuple_element_t<2, T>{}() == TDMGatherIndexSize::Row16bit_Index,
                               uint16_t,
                               uint32_t>;
    };
    using GatherModeDType =
        GatherModeDTypeHelper<TypeParam, std::is_same_v<GatherMode, GatherModeEnable>>::type;

    static constexpr index_t tensor_rank = 2;
    static constexpr index_t tile_m      = 16;
    static constexpr index_t tile_n      = 16;
    static constexpr index_t warp_m      = 1;
    static constexpr index_t warp_n      = 1;
    static constexpr index_t warp_tile_m = 16;
    static constexpr index_t warp_tile_n = 16;

    // Common type definitions
    using TDMShape = TDMTileShape<tensor_rank,
                                  sequence<tile_m, tile_n>,
                                  sequence<warp_m, warp_n>,
                                  sequence<warp_tile_m, warp_tile_n>>;

    // Constants
    static constexpr index_t warp_size     = 32;
    static constexpr index_t cluster_dim_x = 2;
    static constexpr index_t cluster_dim_y = 1;
    static constexpr index_t cluster_dim_z = 1;

    private:
    // Helper functions
    static std::vector<index_t> get_tensor_dims(const TDMTestParams& params, bool is_cluster_test)
    {
        return (!is_cluster_test && std::is_same_v<Layout, tensor_layout::gemm::ColumnMajor>)
                   ? std::vector<index_t>{params.n, params.m}
                   : std::vector<index_t>{params.m, params.n};
    }

    template <bool IsClusterMode, bool IsGatherMode>
    struct TDMTraitsFactory
    {
        using type = TDMPipelineTraits<
            DataType,
            std::conditional_t<IsClusterMode, tensor_layout::gemm::RowMajor, Layout>,
            GatherModeDType,
            false,        /*AtomicBarrierEnable_*/
            IsGatherMode, /*IsGatherMode_*/
            false,        /*IterateEnable_*/
            false,        /*PadEnable_*/
            false,        /*EarlyTimeOutEnable_*/
            IsClusterMode /*ClusterEnable_*/>;
    };

    struct TDMTestData
    {
        HostTensor<DataType> x_host;
        HostTensor<DataType> y_host;
        HostTensor<DataType> ref_host;
        HostTensor<GatherModeDType> gather_index_host;
        DeviceMem x_buf;
        DeviceMem y_buf;
        DeviceMem gather_index_buf;

        TDMTestData(const std::vector<index_t>& dims,
                    const TDMTestParams& params,
                    bool use_cluster,
                    bool use_gather)
            : x_host({dims[0], dims[1]}, {params.x_stride, 1}),
              y_host({dims[0], dims[1]}, {params.y_stride, 1}),
              ref_host({dims[0], dims[1]}, {params.y_stride, 1}),
              gather_index_host(use_gather ? std::vector<index_t>{warp_tile_m}
                                           : std::vector<index_t>{}),
              x_buf(x_host.get_element_space_size_in_bytes()),
              y_buf(y_host.get_element_space_size_in_bytes()),
              gather_index_buf(use_gather ? gather_index_host.get_element_space_size_in_bytes() : 0)
        {
            FillUniformDistribution<DataType>{-.5f, .5f}(x_host);

            if(use_gather)
            {
                for(index_t i = 0; i < warp_tile_m; i++)
                {
                    gather_index_host.data()[i] = static_cast<GatherModeDType>(i);
                }
                std::shuffle(gather_index_host.begin(),
                             gather_index_host.end(),
                             std::mt19937{std::random_device{}()});
                gather_index_buf.ToDevice(gather_index_host.data());

                for(index_t r = 0; r < dims[0]; r += warp_tile_m)
                {
                    for(index_t inner_r = 0; inner_r < warp_tile_m; inner_r++)
                    {
                        index_t ref_idx = 0;
                        index_t gather_idx =
                            static_cast<index_t>(gather_index_host(static_cast<size_t>(inner_r)));
                        for(index_t c = 0; c < dims[1]; c++)
                        {
                            ref_host({static_cast<size_t>(r + inner_r + ref_idx),
                                      static_cast<size_t>(c)}) =
                                x_host(
                                    {static_cast<size_t>(r + gather_idx), static_cast<size_t>(c)});
                        }
                        ref_idx++;
                    }
                }
            }
            else
            {
                for(index_t r = 0; r < dims[0]; r += 1)
                {
                    for(index_t c = 0; c < dims[1]; c += 1)
                    {
                        ref_host({static_cast<size_t>(r), static_cast<size_t>(c)}) =
                            x_host({static_cast<size_t>(r), static_cast<size_t>(c)});
                    }
                }
            }

            if(use_cluster)
            {
                // for sanity test; only copy the fist half data.
                for(index_t r = 0; r < dims[0]; r += 1)
                {
                    for(index_t c = 0; c < dims[1]; c += 1)
                    {
                        ref_host({static_cast<size_t>(r), static_cast<size_t>(c)}) =
                            r >= dims[0] / 2
                                ? x_host({static_cast<size_t>(r - dims[0] / 2),
                                          static_cast<size_t>(c)})
                                : x_host({static_cast<size_t>(r), static_cast<size_t>(c)});
                    }
                }
            }

            x_buf.ToDevice(x_host.data());
            y_buf.SetZero();
        }
    };

    template <typename TDMProblemType>
    bool launch_tdm_kernel(TDMTestData& test_data,
                           const TDMTestParams& params,
                           bool use_cluster = false,
                           bool use_gather  = true)
    {
        dim3 grid((params.m + tile_m - 1) / tile_m, (params.n + tile_n - 1) / tile_n);
        assert(is_wave32());
        const index_t block_size = warp_m * warp_n * warp_size;
        dim3 block(block_size);

        stream_config s{nullptr, false, 0, params.warmup, params.repeat};

        // Determine gather pointer based on usage
        void* gather_ptr = use_gather ? test_data.gather_index_buf.GetDeviceBuffer() : nullptr;

        TDMCopyDeviceKernArgs args{test_data.x_buf.GetDeviceBuffer(),
                                   test_data.y_buf.GetDeviceBuffer(),
                                   gather_ptr,
                                   params.m,
                                   params.n,
                                   params.x_stride,
                                   params.y_stride};

        if(use_cluster)
        {
            hipLaunchConfig_t config{};
            config.gridDim          = grid;
            config.blockDim         = block;
            config.dynamicSmemBytes = 0;
            config.stream           = s.stream_id_;

            hipLaunchAttribute attribute[1];
            attribute[0].id               = hipLaunchAttributeClusterDimension;
            attribute[0].val.clusterDim.x = cluster_dim_x;
            attribute[0].val.clusterDim.y = cluster_dim_y;
            attribute[0].val.clusterDim.z = cluster_dim_z;
            config.attrs                  = attribute;
            config.numAttrs               = 1;

            auto kernel_func = kentry<CK_TILE_MIN_BLOCK_PER_CU,
                                      TDMCopyKernel<TDMProblemType>,
                                      TDMCopyDeviceKernArgs>;
            HIP_CHECK_ERROR(hipLaunchKernelEx(&config, kernel_func, args));
        }
        else
        {
            TDMCopyKernel<TDMProblemType> tdm_kernel;
            launch_kernel(s, make_kernel(tdm_kernel, grid, block, 0, args));
        }

        test_data.y_buf.FromDevice(test_data.y_host.data());
        return true;
    }

    bool validate_results(TDMTestData& test_data) const
    {
        return check_err(
            test_data.y_host, test_data.ref_host, "Error: Incorrect tdm copy results!");
    }

    template <bool IsClusterMode, bool IsGatherMode>
    bool run_tdm_test_generic(const TDMTestParams& params)
    {
        const std::vector<index_t> dims = get_tensor_dims(params, IsClusterMode);
        TDMTestData test_data(dims, params, IsClusterMode, IsGatherMode);

        using TDMTraits  = typename TDMTraitsFactory<IsClusterMode, IsGatherMode>::type;
        using TDMProblem = TDMPipelineProblem<TDMShape, TDMTraits>;

        launch_tdm_kernel<TDMProblem>(test_data, params, IsClusterMode, IsGatherMode);

        if(params.do_validation)
        {
            return validate_results(test_data);
        }

        return true;
    }

    public:
    bool run_tdm_test(const TDMTestParams& params)
    {
        return run_tdm_test_generic<false, std::is_same_v<GatherMode, GatherModeEnable>>(params);
    }

    template <bool is_gather_enable = false>
    bool run_tdm_cluster_test(const TDMTestParams& params)
    {
        return run_tdm_test_generic<true, is_gather_enable>(params);
    }
};

TYPED_TEST_SUITE(TDMBasicTypedTest, TestTypes);

TYPED_TEST(TDMBasicTypedTest, SanityTest)
{
    TDMTestParams params;
    params.m = 16;
    params.n = 16;

    params.template normalize<typename TestFixture::Layout>();

    EXPECT_TRUE(this->run_tdm_test(params));
}

TYPED_TEST(TDMBasicTypedTest, SanityClusterTest)
{
    TDMTestParams params;
    params.m = 32;
    params.n = 16;
    if constexpr(std::is_same_v<typename TestFixture::Layout, Col>)
    {
        GTEST_SKIP();
    }
    params.template normalize<typename TestFixture::Layout>();

    EXPECT_TRUE(this->run_tdm_cluster_test(params));
}

TYPED_TEST(TDMBasicTypedTest, SanityClusterGatherTest)
{
    TDMTestParams params;
    params.m = 32;
    params.n = 16;
    if constexpr(std::is_same_v<typename TestFixture::Layout, Col>)
    {
        GTEST_SKIP();
    }
    params.template normalize<typename TestFixture::Layout>();

    EXPECT_TRUE(this->template run_tdm_cluster_test<true>(params));
}

TYPED_TEST(TDMBasicTypedTest, RectangleTest)
{
    TDMTestParams params;
    params.m = 64;
    params.n = 32;

    params.template normalize<typename TestFixture::Layout>();

    EXPECT_TRUE(this->run_tdm_test(params));
}

TYPED_TEST(TDMBasicTypedTest, LargeDimTest)
{
    TDMTestParams params;
    params.m = 256;
    params.n = 256;

    params.template normalize<typename TestFixture::Layout>();

    EXPECT_TRUE(this->run_tdm_test(params));
}

} // namespace test
} // namespace ck_tile

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
