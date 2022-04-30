#ifndef CK_BLOCKWISE_GEMM_AVX2_HPP
#define CK_BLOCKWISE_GEMM_AVX2_HPP

#include "common_header.hpp"
#include "multi_index_transform_helper.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "threadwise_gemm_avx2.hpp"

namespace ck {
namespace cpu {

template <typename FloatA,
          typename FloatB,
          typename FloatC,

          typename ABlockDesc,
          typename BBlockDesc,
          typename CDesc,

          ck::index_t KPerBlock,

          typename ThreadwiseGemm_Dispatch,
          typename ThreadMNAccessOrder // how we acces gemm MN to utilize micro kernel
          >
struct BlockwiseGemmAvx2_MxN
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    static constexpr index_t nDimA = ABlockDesc::GetNumOfDimension();
    static constexpr index_t nDimB = BBlockDesc::GetNumOfDimension();
    static constexpr index_t nDimC = CDesc::GetNumOfDimension();
    using IndexA                   = MultiIndex<nDimA>;
    using IndexB                   = MultiIndex<nDimB>;
    using IndexC                   = MultiIndex<nDimC>;

    using ACoord = decltype(make_tensor_coordinate(ABlockDesc{}, IndexA{}));
    using BCoord = decltype(make_tensor_coordinate(BBlockDesc{}, IndexB{}));
    using CCoord = decltype(make_tensor_coordinate(CDesc{}, IndexC{}));

    template <typename TensorDesc>
    static constexpr auto GetLeadingElement(const TensorDesc& desc)
    {
        // if use this function, make sure desc are known at compile time.
        // otherwise, it is not efficient to calculate leading dim here
        if constexpr(TensorDesc::GetNumOfDimension() == 1)
        {
            return 1;
        }
        else
        {
            constexpr auto last_dims =
                typename uniform_sequence_gen<TensorDesc::GetNumOfDimension() - 1, 0>::type{};
            constexpr auto lead_dims = decltype(last_dims)::PushFront(Number<1>{});
            return desc.CalculateOffset(lead_dims);
        }
    }

    static ck::index_t GetALeadingElement(const ABlockDesc& a_block_desc)
    {
        return a_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
    }

    static ck::index_t GetBLeadingElement(const BBlockDesc& b_block_desc)
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // K * N
            return b_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
        }
        else
        {
            // N/8 * K * 8
            return b_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}] *
                   b_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<2>{}];
        }
    }

    static ck::index_t GetCLeadingElement(const CDesc& c_desc)
    {
        return c_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
    }

    static ck::index_t GetMPerBlock(const ABlockDesc& a_block_desc)
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // M * K
            return a_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}];
        }
        else
        {
            // K * M
            return a_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
        }
    }

    static ck::index_t GetKPerBlock(const ABlockDesc& a_block_desc)
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // M * K
            return a_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
        }
        else
        {
            // K * M
            return a_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}];
        }
    }

    static ck::index_t GetNPerBlock(const BBlockDesc& b_block_desc)
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // K * N
            return b_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
        }
        else
        {
            // N/8 * K * 8
            return b_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<0>{}] *
                   b_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<2>{}];
        }
    }

    static ck::index_t
    GetABlockStartOffset(const ABlockDesc& a_block_desc, const index_t i_m, const index_t)
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixALayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            return i_m * a_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
        }
        else
        {
            return i_m;
        }
    }

    static ck::index_t
    GetBBlockStartOffset(const BBlockDesc& b_block_desc, const index_t, const index_t i_n)
    {
        if constexpr(std::is_same<typename ThreadwiseGemm_Dispatch::MatrixBLayout,
                                  ck::tensor_layout::gemm::RowMajor>::value)
        {
            // K * N
            return i_n;
        }
        else
        {
            // N/8 * K * 8
            return i_n * b_block_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}];
        }
    }

    static ck::index_t
    GetCBlockStartOffset(const CDesc& c_desc, const index_t i_m, const index_t i_n)
    {
        return i_m * c_desc.GetTransforms()[Number<0>{}].GetUpperLengths()[Number<1>{}] + i_n;
    }

    template <typename ABlockBuffer, typename BBlockBuffer, typename CBuffer>
    static void Run(const ABlockDesc& a_block_desc,
                    const ABlockBuffer& a_block_buf,
                    const IndexA& /* a_origin */,

                    const BBlockDesc& b_block_desc,
                    const BBlockBuffer& b_block_buf,
                    const IndexB& /* b_origin */,

                    const CDesc& c_desc,
                    CBuffer& c_buf,
                    const IndexC& /* c_origin */,

                    bool is_accumulate_c = true)
    {
        auto lda = GetALeadingElement(a_block_desc) * sizeof(FloatA);
        auto ldb = GetBLeadingElement(b_block_desc) * sizeof(FloatB);
        auto ldc = GetCLeadingElement(c_desc) * sizeof(FloatC);

        // printf("lda:%d, ldb:%d, ldc:%d\n", lda, ldb, ldc);

        const auto k_per_block  = GetKPerBlock(a_block_desc);
        const auto m_per_block  = GetMPerBlock(a_block_desc);
        const auto n_per_block  = GetNPerBlock(b_block_desc);
        const auto m_per_thread = ThreadwiseGemm_Dispatch::ThreadMaxMr;
        const auto n_per_thread = ThreadwiseGemm_Dispatch::ThreadMaxNr;

        ck::cpu::ThreadwiseGemmParam param;
        param.Kr          = k_per_block;
        param.lda         = lda;
        param.ldb         = ldb;
        param.ldc         = ldc;
        param.alpha       = 1.0f; // TODO
        param.accmulate_c = is_accumulate_c ? 1 : 0;

        if constexpr(std::is_same<ThreadMNAccessOrder, ck::Sequence<0, 1>>::value)
        {
            for(ck::index_t i_m = 0; i_m < m_per_block; i_m += m_per_thread)
            {
                auto current_mr = ck::math::min(m_per_block - i_m, m_per_thread);
                param.p_a       = &a_block_buf.p_data_[GetABlockStartOffset(a_block_desc, i_m, 0)];

                // printf("YYYY: %d, i_m:%d, current_mr:%d, %d, %p\n",__LINE__, i_m, current_mr,
                // GetABlockStartOffset(a_block_desc, i_m, 0), param.p_a);fflush(stdout);

                for(ck::index_t i_n = 0; i_n < n_per_block; i_n += n_per_thread)
                {
                    auto current_nr = ck::math::min(n_per_block - i_n, n_per_thread);

                    param.p_b = &b_block_buf.p_data_[GetBBlockStartOffset(b_block_desc, 0, i_n)];
                    param.p_c = &c_buf.p_data_[GetCBlockStartOffset(c_desc, i_m, i_n)];

                    // printf("YYYY: %d, i_n:%d, current_nr:%d, %d, %p, C:%d, %p\n",__LINE__, i_n,
                    // current_nr, GetBBlockStartOffset(b_block_desc, 0, i_n), param.p_b,
                    //        GetCBlockStartOffset(c_desc, i_m, i_n),
                    //        param.p_c);fflush(stdout);

                    ThreadwiseGemm_Dispatch::Run(&param, current_mr, current_nr);
                }
            }
        }
    }
};

} // namespace cpu
} // namespace ck
#endif
