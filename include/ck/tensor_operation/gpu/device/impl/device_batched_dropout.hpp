// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>
#include <numeric>

#include "ck/utility/common_header.hpp"
#include "ck/utility/philox_rand.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/device_base.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/masking_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_batched_dropout.hpp"
#include "ck/tensor_operation/operator_transform/transform_contraction_to_gemm.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename GridwiseDropout_,
          typename ZDataType,
          typename ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3,
          typename Block2CTileMap,
          typename ComputeBasePtrOfStridedBatch>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_dropout(ZDataType* __restrict__ p_z_grid,
                               const ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3
                                   c_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                               const Block2CTileMap block_2_ctile_map,
                               const index_t num_gemm0_m_block_outer_loop,
                               const index_t batch_count,
                               const ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch,
                               const unsigned long long seed,
                               const unsigned long long offset,
                               const index_t raw_m_padded,
                               const index_t raw_n_padded)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t z_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_base_ptr_of_batch.GetZBasePtr(g_idx)));

    ck::philox ph(seed, 0, offset);
    ZDataType* z_matrix_ptr = (p_z_grid == nullptr ? nullptr : p_z_grid + z_batch_offset);

    const index_t z_random_matrix_offset = g_idx * raw_m_padded * raw_n_padded;

    GridwiseDropout_::Run(z_matrix_ptr,
                          c_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3,
                          block_2_ctile_map,
                          ph,
                          num_gemm0_m_block_outer_loop,
                          z_random_matrix_offset,
                          raw_n_padded);
#else
    ignore = p_z_grid;
    ignore = c_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3;
    ignore = block_2_ctile_map;
    ignore = num_gemm0_m_block_outer_loop;
    ignore = batch_count;
    ignore = compute_base_ptr_of_batch;
    ignore = seed;
    ignore = offset;
    ignore = raw_m_padded;
    ignore = raw_n_padded;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

// Computes C = A * B0 * B1
//              ^^^^^^ (Acc0)
//              ^^^^^^^^^^^ (Acc1)
template <index_t NumDimG,
          typename GemmDataType,
          typename ZDataType,
          typename GemmAccDataType,
          GemmSpecialization GemmSpec,
          TensorSpecialization ASpec,
          TensorSpecialization BSpec,
          TensorSpecialization B1Spec,
          TensorSpecialization CSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock, // Gemm0NPerBlock
          index_t KPerBlock, // Gemm0KPerBlock
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave>
struct DeviceBatchedDropout : public ck::tensor_operation::device::BaseOperator
{
    static_assert(NumDimG > 0, "Number of dimension must be greater than 0");

    using DeviceOp = DeviceBatchedDropout;

    static constexpr index_t Gemm1NPerBlock = 128;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    using Transform = TransformBatchedContractionContractionToBatchedGemmGemm<
        Sequence<NumDimG, 1, 1, 1, 1>, // NumDimM, NumDimN, NumDimK, NumDimO
        Sequence<MPerBlock, NPerBlock, KPerBlock, Gemm1NPerBlock>,
        GemmSpec,
        ASpec,
        BSpec,
        B1Spec,
        CSpec>;

    // Q in Gemm A position
    static auto MakeAGridDescriptor_AK0_M_AK1(const std::vector<index_t>& a_gs_m_k_lengths,
                                              const std::vector<index_t>& a_gs_m_k_strides)
    {
        return Transform::MakeAGridDescriptor_AK0_M_AK1(
            Transform::MakeAGridDescriptor_M_K(a_gs_m_k_lengths, a_gs_m_k_strides), Number<AK1>{});
    }

    // Z in Gemm0 C position
    static auto MakeZGridDescriptor_M_N(const std::vector<index_t>& z_gs_m_n_lengths,
                                        const std::vector<index_t>& z_gs_m_n_strides)
    {
        return Transform::MakeC0GridDescriptor_M_N(z_gs_m_n_lengths, z_gs_m_n_strides);
    }

    using ZGridDesc_G_M_N = decltype(Transform::MakeCGridDescriptor_G_M_N({}, {}));

    using KGridDesc_N_K = decltype(Transform::MakeB0GridDescriptor_N_K({}, {}));
    using ZGridDesc_M_N = decltype(MakeZGridDescriptor_M_N({}, {}));

    struct ComputeBasePtrOfStridedBatch
    {
        ComputeBasePtrOfStridedBatch() {}
        ComputeBasePtrOfStridedBatch(const ZGridDesc_G_M_N& z_grid_desc_g_m_n)
            : z_grid_desc_g_m_n_(z_grid_desc_g_m_n)
        {
        }

        __host__ __device__ constexpr long_index_t GetZBasePtr(index_t g_idx) const
        {
            return z_grid_desc_g_m_n_.CalculateOffset(make_multi_index(g_idx, 0, 0));
        }

        private:
        ZGridDesc_G_M_N z_grid_desc_g_m_n_;
    };

    using GridwiseDropout = GridwiseBatchedDropout<ZDataType,
                                                   GemmDataType,
                                                   GemmAccDataType,
                                                   KGridDesc_N_K,
                                                   ZGridDesc_M_N,
                                                   BlockSize,
                                                   MPerBlock,
                                                   NPerBlock,
                                                   KPerBlock,
                                                   Gemm1NPerBlock,
                                                   AK1,
                                                   BK1,
                                                   MPerXDL,
                                                   NPerXDL,
                                                   MXdlPerWave,
                                                   NXdlPerWave>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(ZDataType* p_z_grid,
                 const std::vector<index_t>& b_gs_n_k_lengths,
                 const std::vector<index_t>& b_gs_n_k_strides,
                 const std::vector<index_t>& z_gs_m_n_lengths,
                 const std::vector<index_t>& z_gs_m_n_strides,
                 std::tuple<unsigned long long, unsigned long long> seeds)
            : p_z_grid_{p_z_grid},
              z_grid_desc_m_n_{MakeZGridDescriptor_M_N(z_gs_m_n_lengths, z_gs_m_n_strides)},
              k_grid_desc_n_k_{
                  Transform::MakeB0GridDescriptor_N_K(b_gs_n_k_lengths, b_gs_n_k_strides)},
              z_grid_desc_g_m_n_{
                  Transform::MakeCGridDescriptor_G_M_N(z_gs_m_n_lengths, z_gs_m_n_strides)},
              block_2_ctile_map_{GridwiseDropout::MakeDefaultBlock2CTileMap(k_grid_desc_n_k_)},
              batch_count_{z_grid_desc_g_m_n_.GetLength(I0)}
        {

            compute_base_ptr_of_batch_ = ComputeBasePtrOfStridedBatch(z_grid_desc_g_m_n_);

            seed_   = std::get<0>(seeds);
            offset_ = std::get<1>(seeds);

            z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_ =
                GridwiseDropout::MakeCGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3(
                    z_grid_desc_m_n_);

            auto m_raw = z_gs_m_n_lengths[NumDimG];
            auto n_raw = z_gs_m_n_lengths[NumDimG + 1];

            m_raw_padded_ = GridwiseDropout::GetPaddedSize(m_raw);
            n_raw_padded_ = GridwiseDropout::GetPaddedSize(n_raw);

            std::vector<index_t> a_gs_m_k_strides(NumDimG + 2, 0);
            std::vector<index_t> a_gs_m_k_lengths = z_gs_m_n_lengths;

            a_gs_m_k_lengths[NumDimG + 1] = 1;

            auto a_grid_desc_k0_m_k1 =
                DeviceOp::MakeAGridDescriptor_AK0_M_AK1(a_gs_m_k_lengths, a_gs_m_k_strides);

            num_gemm0_m_block_outer_loop_ = a_grid_desc_k0_m_k1.GetLength(I1) / MPerBlock;
        }

        // pointers
        ZDataType* p_z_grid_;

        // tensor descriptor
        ZGridDesc_M_N z_grid_desc_m_n_;
        KGridDesc_N_K k_grid_desc_n_k_;

        // batch offsets
        ZGridDesc_G_M_N z_grid_desc_g_m_n_;

        typename GridwiseDropout::ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3
            z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_;

        // block-to-c-tile map
        typename GridwiseDropout::DefaultBlock2CTileMap block_2_ctile_map_;

        index_t num_gemm0_m_block_outer_loop_;

        index_t batch_count_;
        ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch_;

        unsigned long long seed_;
        unsigned long long offset_;

        index_t m_raw_padded_;
        index_t n_raw_padded_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(!DeviceOp::IsSupportedArgument(arg))
            {
                throw std::runtime_error("wrong! unsupported argument");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.k_grid_desc_n_k_) * arg.batch_count_;

            float ave_time = 0;

            auto launch_kernel = [&]() {
                const auto kernel = kernel_batched_dropout<
                    GridwiseDropout,
                    ZDataType,
                    typename GridwiseDropout::ZGridDescriptor_M0_N0_M1_N1_M2_N2_M3_M4_M5_N3,
                    typename GridwiseDropout::DefaultBlock2CTileMap,
                    ComputeBasePtrOfStridedBatch>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_z_grid_,
                                              arg.z_grid_desc_m0_n0_m1_n1_m2_n2_m3_m4_m5_n3_,
                                              arg.block_2_ctile_map_,
                                              arg.num_gemm0_m_block_outer_loop_,
                                              arg.batch_count_,
                                              arg.compute_base_ptr_of_batch_,
                                              arg.seed_,
                                              arg.offset_,
                                              arg.m_raw_padded_,
                                              arg.n_raw_padded_);
            };
            ave_time = launch_kernel();
            return ave_time;
        }

        // polymorphic
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
        (void)arg;

        if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
             ck::get_device_name() == "gfx940" || ck::get_device_name() == "gfx941" ||
             ck::get_device_name() == "gfx942"))
        {
            return false;
        }

        return GridwiseDropout::CheckValidity();
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(ZDataType* p_z,
                             const std::vector<index_t>& z_gs_m_n_lengths,
                             const std::vector<index_t>& z_gs_m_n_strides,
                             std::tuple<unsigned long long, unsigned long long> seeds)
    {
        std::vector<index_t> b_gs_n_k_strides(NumDimG + 2, 0);
        std::vector<index_t> b_gs_n_k_lengths = z_gs_m_n_lengths;

        b_gs_n_k_lengths[NumDimG]     = z_gs_m_n_lengths[NumDimG + 1];
        b_gs_n_k_lengths[NumDimG + 1] = 1;

        return Argument{
            p_z, b_gs_n_k_lengths, b_gs_n_k_strides, z_gs_m_n_lengths, z_gs_m_n_strides, seeds};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(void* p_z,
                        const std::vector<index_t>& z_gs_m_n_lengths,
                        const std::vector<index_t>& z_gs_m_n_strides,
                        std::tuple<unsigned long long, unsigned long long> seeds) // override
    {
        std::vector<index_t> b_gs_n_k_strides(NumDimG + 2, 0);
        std::vector<index_t> b_gs_n_k_lengths = z_gs_m_n_lengths;

        b_gs_n_k_lengths[NumDimG]     = z_gs_m_n_lengths[NumDimG + 1];
        b_gs_n_k_lengths[NumDimG + 1] = 1;

        return std::make_unique<Argument>(static_cast<ZDataType*>(p_z),
                                          b_gs_n_k_lengths,
                                          b_gs_n_k_strides,
                                          z_gs_m_n_lengths,
                                          z_gs_m_n_strides,
                                          seeds);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() // override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchedDropout"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << AK1 << ", "
            << BK1 << ", "
            << MPerBlock << ", "
            << Gemm1NPerBlock << ", "
            << getGemmSpecializationString(GemmSpec) << ", "
            << "ASpec" << getTensorSpecializationString(ASpec) << ", "
            << "B0Spec" << getTensorSpecializationString(BSpec) << ", "
            << "B1Spec" << getTensorSpecializationString(B1Spec) << ", "
            << "CSpec" << getTensorSpecializationString(CSpec);
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
