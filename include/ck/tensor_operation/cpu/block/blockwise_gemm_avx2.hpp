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
          typename AccDataType,
          typename ABlockDesc,
          typename BBlockDesc,
          typename CBlockDesc,

          typename ABlockSliceLengths,
          typename BBlockSliceLengths,
          typename CBlockSliceLengths,

          typename AThreadSliceLength,
          typename BThreadSliceLength,

          ck::index_t AThreadLoopOverDim, // thread slice loop over on block slice. 1d is enough for
                                          // now
          ck::index_t BThreadLoopOverDim,

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
    static constexpr index_t nDimC = CBlockDesc::GetNumOfDimension();
    using IndexA                   = MultiIndex<nDimA>;
    using IndexB                   = MultiIndex<nDimB>;
    using IndexC                   = MultiIndex<nDimC>;

    using ACoord = decltype(make_tensor_coordinate(ABlockDesc{}, IndexA{}));
    using BCoord = decltype(make_tensor_coordinate(BBlockDesc{}, IndexB{}));
    using CCoord = decltype(make_tensor_coordinate(CBlockDesc{}, IndexC{}));

#if 0
    constexpr BlockwiseGemmAvx2_MxN(const ABlockDesc & a_block_desc, const IndexA& a_thread_origin,
                                    const BBlockDesc & b_block_desc, const IndexB& b_thread_origin)
        : a_thread_coord_(make_tensor_coordinate(a_block_desc, a_thread_origin)),
          b_thread_coord_(make_tensor_coordinate(b_block_desc, b_thread_origin)),
    {

    }
#endif

    template <typename TensorDesc>
    constexpr auto GetLeadingElement(const TensorDesc& desc)
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

    template <typename ABlockBuffer, typename BBlockBuffer, typename CBlockBuffer>
    void Run(const ABlockDesc& a_block_desc,
             const ABlockBuffer& a_block_buf,
             const IndexA& a_origin,

             const BBlockDesc& b_block_desc,
             const BBlockBuffer& b_block_buf,
             const IndexB& b_origin,

             const CBlockDesc& c_block_desc,
             CBlockBuffer& c_block_buf,
             const IndexC& c_origin) const
    {

        constexpr auto m_n_block_length =
            ck::Sequence<ABlockSliceLengths::At(AThreadLoopOverDim),
                         BBlockSliceLengths::At(BThreadLoopOverDim)>{};
        constexpr auto m_n_thread_length =
            ck::Sequence<AThreadSliceLength::At(AThreadLoopOverDim),
                         BThreadSliceLength::At(BThreadLoopOverDim)>{};

        constexpr auto m_n_access_length = m_n_block_length / m_n_thread_length;

        constexpr auto ordered_m_n_access_length =
            container_reorder_given_new2old(m_n_access_length, ThreadMNAccessOrder{});

        constexpr auto a_block_idx_zeros =
            typename uniform_sequence_gen<nDimA, 0>::type{}; // starting point of the block
        constexpr auto b_block_idx_zeros = typename uniform_sequence_gen<nDimB, 0>::type{};

        constexpr auto lda = GetLeadingElement(a_block_desc) * sizeof(FloatA);
        constexpr auto ldb = GetLeadingElement(b_block_desc) * sizeof(FloatB);
        constexpr auto ldc = GetLeadingElement(c_block_desc) * sizeof(FloatC);

        ck::cpu::ThreadwiseGemmParam param;
        param.Kr    = KPerBlock;
        param.lda   = lda;
        param.ldb   = ldb;
        param.ldc   = ldc;
        param.alpha = 1.0f; // TODO

        static_ford<decltype(ordered_m_n_access_length)>{}([&](auto ordered_idx) {
            constexpr auto origin_m_n_idx = ordered_idx.ReorderGivenOld2New(ThreadMNAccessOrder{});

            constexpr auto current_m_idx =
                origin_m_n_idx.At(0) * AThreadSliceLength::At(AThreadLoopOverDim);
            constexpr auto current_n_idx =
                origin_m_n_idx.At(1) * BThreadSliceLength::At(BThreadLoopOverDim);

            constexpr auto current_mr =
                ck::math::min(m_n_block_length.At(0) - current_m_idx, m_n_thread_length.At(0));
            constexpr auto current_nr =
                ck::math::min(m_n_block_length.At(1) - current_n_idx, m_n_thread_length.At(1));

            constexpr auto a_block_idx =
                a_block_idx_zeros.Modify(AThreadLoopOverDim, current_m_idx);
            constexpr auto a_block_coord =
                make_tensor_coordinate(a_block_desc, to_multi_index(a_origin + a_block_idx));

            constexpr auto b_block_idx =
                b_block_idx_zeros.Modify(BThreadLoopOverDim, current_n_idx);
            constexpr auto b_block_coord =
                make_tensor_coordinate(b_block_desc, to_multi_index(b_origin + b_block_idx));

            constexpr auto c_block_coord =
                make_tensor_coordinate(c_block_desc, to_multi_index(c_origin + origin_m_n_idx));

            param.p_a = &a_block_buf.p_data_[a_block_coord.GetOffset()];
            param.p_b = &b_block_buf.p_data_[b_block_coord.GetOffset()];
            param.p_c = &c_block_buf.p_data_[c_block_coord.GetOffset()];

            ThreadwiseGemm_Dispatch::Run(&param, current_mr, current_nr);
        });
    }
};

} // namespace cpu
} // namespace ck
#endif
