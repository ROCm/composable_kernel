#ifndef CK_XDLOPS_GEMM_HPP
#define CK_XDLOPS_GEMM_HPP

#include "common_header.hpp"
#include "math.hpp"
#include "amd_xdlops.hpp"

namespace ck {

enum struct mfma_instr
{
    mfma_f32_32x32x1xf32 = 0,
    mfma_f32_16x16x1xf32,
    mfma_f32_4x4x1xf32,
    mfma_f32_32x32x2xf32, // k reduction
    mfma_f32_16x16x4xf32, // k reduction
    mfma_f32_32x32x4f16,
    mfma_f32_16x16x4f16,
    mfma_f32_4x4x4f16,
    mfma_f32_32x32x8f16,  // k reduction
    mfma_f32_16x16x16f16, // k reduction
    mfma_f32_32x32x2bf16,
    mfma_f32_16x16x2bf16,
    mfma_f32_4x4x2bf16,
    mfma_f32_32x32x4bf16, // k reduction
    mfma_f32_16x16x8bf16, // k reduction
};

template <mfma_instr instr>
struct mfma_info;

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x1xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x1f32<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x2xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x2f32<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x4xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x4f32<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x1xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x1f32<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

// treat 4x4x1 as a single-blk 4x64 mfma
template <>
struct mfma_info<mfma_instr::mfma_f32_4x4x1xf32>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 1;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_4x4x1f32<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x4f16<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x8f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_32x32x8f16<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x16f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x16f16<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_16x16x4f16<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_4x4x4f16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 4;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t COffset,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ void run(const FloatA& a, const FloatB& b, FloatC& reg_c) const
    {
        intrin_mfma_f32_4x4x4f16<MPerXdlops, NPerXdlops, COffset>::Run(a, b, reg_c);
    }
};

#if 0
template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x2bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 2;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_32x32x2bf16<MPerXdlops, NPerXdlops, AStride, BStride>::run(
            p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_32x32x4bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 4;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 32;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 32;
    static constexpr index_t n_per_blk           = 32;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_32x32x4bf16(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x8bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = true;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_16x16x8bf16(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_16x16x2bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 16;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = wave_size / num_threads_per_blk;
    static constexpr index_t num_output_blks     = 4;
    static constexpr index_t m_per_blk           = 16;
    static constexpr index_t n_per_blk           = 16;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_16x16x2bf16<MPerXdlops, NPerXdlops>(p_a, p_b, reg_c);
    }
};

template <>
struct mfma_info<mfma_instr::mfma_f32_4x4x2bf16>
{
    static constexpr index_t group_size          = 4;
    static constexpr index_t num_groups_per_blk  = 1;
    static constexpr index_t num_regs_per_blk    = group_size * num_groups_per_blk;
    static constexpr index_t num_threads_per_blk = 64;
    static constexpr index_t wave_size           = 64;
    static constexpr index_t num_input_blks      = 1;
    static constexpr index_t num_output_blks     = 1;
    static constexpr index_t m_per_blk           = 4;
    static constexpr index_t n_per_blk           = 64;
    static constexpr index_t k_per_blk           = 2;
    static constexpr bool is_k_reduction         = false;

    template <index_t MPerXdlops,
              index_t NPerXdlops,
              index_t AStride,
              index_t BStride,
              class FloatA,
              class FloatB,
              class FloatC>
    __device__ FloatC run(const FloatA* a, const FloatB* b, FloatC reg_c) const
    {
        const auto p_a = c_style_pointer_cast<const ushort2_t*>(a);
        const auto p_b = c_style_pointer_cast<const ushort2_t*>(b);

        return intrin_mfma_f32_4x4x2bf16<MPerXdlops, NPerXdlops>::run(p_a, p_b, reg_c);
    }
};
#endif

template <mfma_instr instr, index_t MPerXdlops_, index_t NPerXdlops_>
struct xdlops_info
{
    static constexpr auto mfma_type = mfma_info<instr>{};

    static constexpr index_t MPerXdlops = MPerXdlops_;
    static constexpr index_t NPerXdlops = NPerXdlops_;

    static constexpr bool IsABroadcast()
    {
        static_assert(NPerXdlops >= MPerXdlops, "only support ABroadcast");
        return true;
    }

    static constexpr index_t GetKPerXdlops()
    {
        return mfma_type.is_k_reduction ? mfma_type.num_input_blks : 1;
    }

    static constexpr index_t GetNumCRegs() { return MPerXdlops * NPerXdlops / mfma_type.wave_size; }
};

template <class base_type, index_t MPerWave, index_t NPerWave, index_t KPack>
struct XdlopsGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    template <class base_type_  = base_type,
              index_t MPerWave_ = MPerWave,
              index_t NPerWave_ = NPerWave>
    static constexpr auto GetXdlopsInfo();

    template <>
    static constexpr auto GetXdlopsInfo<float, 64, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x1xf32, 64, 64>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 32, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x1xf32, 32, 64>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 16, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x1xf32, 16, 64>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 8, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x1xf32, 8, 64>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 4, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x1xf32, 4, 64>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 32, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2xf32, 32, 32>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<float, 16, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x4xf32, 16, 16>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 64, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x4f16, 64, 64>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 32, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x4f16, 32, 64>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 32, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x8f16, 32, 32>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 16, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x16f16, 16, 16>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 16, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x4f16, 16, 64>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 8, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x4f16, 8, 64>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<half_t, 4, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x4f16, 4, 64>{};
    }

#if 0
    template <>
    static constexpr auto GetXdlopsInfo<ushort, 128, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 64, 64, 2, 1, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 64, 128>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 64, 64, 1, 2, c_vec32_4_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 64, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 64, 64, 1, 1, c_vec32_2_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 64, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 64, 32, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 32, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x2bf16, 32, 64, 1, 1, c_vec32_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 64, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x2bf16, 64, 16, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 16, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x2bf16, 16, 64, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 8, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x2bf16, 8, 64, 1, 1, c_vec4_2_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 4, 64>()
    {
        return xdlops_info<mfma_instr::mfma_f32_4x4x2bf16, 4, 64, 1, 1, c_vec4_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 32, 32>()
    {
        return xdlops_info<mfma_instr::mfma_f32_32x32x4bf16, 32, 32, 1, 1, c_vec16_1_t>{};
    }

    template <>
    static constexpr auto GetXdlopsInfo<ushort, 16, 16>()
    {
        return xdlops_info<mfma_instr::mfma_f32_16x16x8bf16, 16, 16, 1, 1, c_vec4_1_t>{};
    }
#endif

    using CIndex = MultiIndex<2>;

    __device__ static constexpr index_t GetNumBlks() { return mfma_type.num_output_blks; }

    __device__ static constexpr index_t GetNumXdlops()
    {
        return MPerXdlops * NPerXdlops /
               (mfma_type.m_per_blk * mfma_type.n_per_blk * mfma_type.num_output_blks);
    }

    __host__ __device__ static void mfma_info_check()
    {

        static_assert(mfma_type.num_threads_per_blk == mfma_type.n_per_blk,
                      "n_per_blk != num_threads_per_blk");

        static_assert(mfma_type.num_regs_per_blk * mfma_type.num_input_blks == mfma_type.m_per_blk,
                      "m_per_blk != num_input_blks * num_regs_per_blk");

        static_assert(mfma_type.num_output_blks == mfma_type.num_input_blks ||
                          mfma_type.num_output_blks == 1,
                      "incorrect num_output_blks");

        static_assert(mfma_type.num_regs_per_blk * mfma_type.wave_size ==
                          mfma_type.m_per_blk * mfma_type.n_per_blk,
                      "num_regs_per_blk incorrect");

        static_assert(mfma_type.is_k_reduction ||
                          (mfma_type.num_input_blks == mfma_type.num_output_blks),
                      "is_k_reduction wrong!");
    }

    __host__ __device__ constexpr XdlopsGemm()
    {
        static_assert(NPerXdlops == 4 || NPerXdlops == 8 || NPerXdlops == 16 || NPerXdlops == 32 ||
                          NPerXdlops == 64,
                      "Only support GemmNPerXdlops == 4, 8, 16, 32 or 64 for xdlops");

        static_assert(MPerXdlops == 4 || MPerXdlops == 8 || MPerXdlops == 16 || MPerXdlops == 32 ||
                          MPerXdlops == 64,
                      "Only support GemmMPerXdlops == 4, 8, 16, 32 or 64 for xdlops");
    }

    template <typename CM0N0M1N1M2N2GridDesc>
    __host__ __device__ static constexpr auto
    MakeCM0N0M1N1M2M3M4N2GridDescriptor(const CM0N0M1N1M2N2GridDesc& c_m0_n0_m1_n1_m2_n2_grid_desc)
    {
        constexpr auto M0 = c_m0_n0_m1_n1_m2_n2_grid_desc.GetLength(I0);
        constexpr auto N0 = c_m0_n0_m1_n1_m2_n2_grid_desc.GetLength(I1);
        constexpr auto M1 = c_m0_n0_m1_n1_m2_n2_grid_desc.GetLength(I2);
        constexpr auto N1 = c_m0_n0_m1_n1_m2_n2_grid_desc.GetLength(I3);
        constexpr auto M2 = c_m0_n0_m1_n1_m2_n2_grid_desc.GetLength(I4);
        constexpr auto N2 = c_m0_n0_m1_n1_m2_n2_grid_desc.GetLength(I5);

        static_assert(N2 == mfma_type.num_threads_per_blk, "");
        static_assert(
            M2 == (mfma_type.num_groups_per_blk * mfma_type.num_output_blks * mfma_type.group_size),
            "");

        return transform_dynamic_tensor_descriptor(
            c_m0_n0_m1_n1_m2_n2_grid_desc,
            make_tuple(make_pass_through_transform(M0),
                       make_pass_through_transform(N0),
                       make_pass_through_transform(M1),
                       make_pass_through_transform(N1),
                       make_unmerge_transform(make_tuple(mfma_type.num_groups_per_blk,
                                                         mfma_type.num_input_blks,
                                                         mfma_type.group_size)),
                       make_pass_through_transform(mfma_type.num_threads_per_blk)),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4>{},
                       Sequence<5>{}),
            make_tuple(Sequence<0>{},
                       Sequence<1>{},
                       Sequence<2>{},
                       Sequence<3>{},
                       Sequence<4, 5, 6>{},
                       Sequence<7>{}));
    }

    __device__ static constexpr index_t GetRegSizePerXdlops()
    {
        return MPerXdlops * NPerXdlops / mfma_type.wave_size;
    }

    template <index_t c_offset, class FloatA, class FloatB, class FloatC>
    __device__ void Run(const FloatA& p_a_wave, const FloatB& p_b_wave, FloatC& p_c_thread) const
    {
        static_assert(is_same<base_type, float>::value || is_same<base_type, half_t>::value ||
                          is_same<base_type, ushort>::value,
                      "base base_type must be float, half, ushort!");

        static_assert(KPack % mfma_type.k_per_blk == 0, "KPack cannot be divided by k_per_blk");

        static_for<0, KPack / mfma_type.k_per_blk, 1>{}([&](auto k) {
            mfma_type.template run<MPerXdlops, NPerXdlops, c_offset>(
                p_a_wave[k], p_b_wave[k], p_c_thread);
        });
    }

    __device__ static auto GetLaneId() { return get_thread_local_1d_id() % mfma_type.wave_size; }

    __device__ static auto GetBlkIdx(const index_t laneId)
    {
        const auto threadidx_to_blk_idx_adaptor = make_single_stage_tensor_adaptor(
            make_tuple(make_merge_transform(
                make_tuple(1, mfma_type.num_input_blks, mfma_type.num_threads_per_blk))),
            make_tuple(Sequence<0, 1, 2>{}),
            make_tuple(Sequence<0>{}));

        const auto blk_idx =
            threadidx_to_blk_idx_adaptor.CalculateBottomIndex(make_multi_index(laneId));

        const auto blk_id = blk_idx[I1];
        const auto blk_td = blk_idx[I2];

        return make_tuple(blk_id, blk_td);
    }

    __host__ __device__ static auto CalculateAThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx(laneId);

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(mfma_type.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __host__ __device__ static auto CalculateBThreadOriginDataIndex()
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx(laneId);

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        if constexpr(mfma_type.is_k_reduction)
        {
            return make_tuple(blk_id, blk_td);
        }
        else
        {
            return make_tuple(0, laneId);
        }
    }

    __device__ static CIndex GetBeginOfThreadBlk(index_t xdlops_i, index_t blk_i)
    {
        const auto laneId  = GetLaneId();
        const auto blk_idx = GetBlkIdx(laneId);

        const auto blk_id = blk_idx[I0];
        const auto blk_td = blk_idx[I1];

        index_t n_offset = blk_i * mfma_type.n_per_blk + blk_td;
        index_t m_offset = xdlops_i * mfma_type.m_per_blk + blk_id * mfma_type.group_size;

        return CIndex{m_offset, n_offset};
    }

    static constexpr index_t MPerXdlops = GetXdlopsInfo().MPerXdlops;
    static constexpr index_t NPerXdlops = GetXdlopsInfo().NPerXdlops;
    static constexpr index_t KPerXdlops = GetXdlopsInfo().GetKPerXdlops();

    static constexpr bool IsABroadcast = GetXdlopsInfo().IsABroadcast();

    static constexpr auto mfma_type = GetXdlopsInfo().mfma_type;

    struct CLayout
    {
        __host__ __device__ static constexpr index_t M1() { return mfma_type.num_groups_per_blk; }
        __host__ __device__ static constexpr index_t M0() { return mfma_type.group_size; }
        __host__ __device__ static constexpr index_t N1() { return mfma_type.num_input_blks; }
        __host__ __device__ static constexpr index_t N0() { return mfma_type.num_threads_per_blk; }

        __device__ static constexpr index_t GetBlkSize() { return mfma_type.num_regs_per_blk; }

        __device__ static constexpr index_t GetNumBlks() { return mfma_type.num_output_blks; }

        __device__ static constexpr index_t GetNumXdlops()
        {
            return MPerXdlops * NPerXdlops /
                   (mfma_type.m_per_blk * mfma_type.n_per_blk * mfma_type.num_output_blks);
        }
    };

    __host__ __device__ static constexpr auto GetCXdlopsLayout() { return CLayout{}; }
};

} // namespace ck
#endif
