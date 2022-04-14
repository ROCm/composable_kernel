#include <iostream>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <tuple>
#include <memory>
#include <half.hpp>
#include <omp.h>
#include "host_tensor.hpp"
#include "tensor_layout.hpp"
#include "device.hpp"
#include "config.hpp"
#include "print.hpp"
#include "cpuid.hpp"
#include "threadwise_tensor_slice_transfer_avx2.hpp"
#include "element_wise_operation_cpu.hpp"
#include "dynamic_buffer_cpu.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

// using AType = half_float::half;
// using BType = half_float::half;
using AType = float;
using BType = float;
using CType = float;
#define NTStore false

using PassThrough = ck::tensor_operation::cpu::element_wise::PassThrough;

static inline int conv_out_size(int in_size, int pad, int dilation, int ksize, int stride)
{
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

#define MC 16
#define NC 24
#define KC 32

#define IsInputPadded true
#define IsInputCBlockTranspose false
#define CBlockMVector 8

template <typename T>
static inline void dump_memory(T* ptr, ck::index_t elem)
{
    for(ck::index_t i = 0; i < elem; i++)
    {
        std::cout << i << ": 0x" << std::hex << ptr[i] << std::dec << std::endl;
    }
}

int main(int argc, char** argv)
{
    int n  = 2;
    int hi = 8;
    int wi = 6;
    int c  = 8;
    int fy = 3;
    int fx = 3;
    int dy = 1;
    int dx = 1;
    int sy = 1;
    int sx = 1;
    int py = 0;
    int px = 0;

    if(argc > 12)
    {
        n  = std::atoi(argv[1]);
        hi = std::atoi(argv[2]);
        wi = std::atoi(argv[3]);
        c  = std::atoi(argv[4]);
        fy = std::atoi(argv[5]);
        fx = std::atoi(argv[6]);
        dy = std::atoi(argv[7]);
        dx = std::atoi(argv[8]);
        sy = std::atoi(argv[9]);
        sx = std::atoi(argv[10]);
        py = std::atoi(argv[11]);
        px = std::atoi(argv[12]);
    }

    int ho = conv_out_size(hi, py, dy, fy, sy);
    int wo = conv_out_size(wi, px, dx, fx, sx);

    DeviceAlignedMemCPU input_mem(n * c * hi * wi * sizeof(AType), 32);
    DeviceAlignedMemCPU input_cblock_mem(MC * KC * sizeof(AType), 32);

    auto gen_input_buffer =
        [&](AType* ptr, ck::index_t N, ck::index_t Hi, ck::index_t Wi, ck::index_t C) {
            for(auto i_n = 0; i_n < N; i_n++)
            {
                for(auto i_hi = 0; i_hi < Hi; i_hi++)
                {
                    for(auto i_wi = 0; i_wi < Wi; i_wi++)
                    {
                        for(auto i_c = 0; i_c < C; i_c++)
                        {
                            auto index = i_n * Hi * Wi * C + i_hi * Wi * C + i_wi * C + i_c;
                            auto value = ((i_n & 0xff) << 24) | ((i_hi & 0xff) << 16) |
                                         ((i_wi & 0xff) << 8) | ((i_c & 0xff) << 0);
                            ptr[index] = *reinterpret_cast<AType*>(&value);
                        }
                    }
                }
            }
        };

    gen_input_buffer(reinterpret_cast<AType*>(input_mem.mpDeviceBuf), n, hi, wi, c);

    const auto input_desc = [&]() {
        const auto in_n_hi_wi_c_grid_desc =
            ck::make_naive_tensor_descriptor_packed(ck::make_tuple(n, hi, wi, c));

        const auto in_n_hip_wip_c_grid_desc = ck::transform_tensor_descriptor(
            in_n_hi_wi_c_grid_desc,
            ck::make_tuple(ck::make_pass_through_transform(n),
                           ck::make_pad_transform(hi, py, py),
                           ck::make_pad_transform(wi, px, px),
                           ck::make_pass_through_transform(c)),
            ck::make_tuple(
                ck::Sequence<0>{}, ck::Sequence<1>{}, ck::Sequence<2>{}, ck::Sequence<3>{}),
            ck::make_tuple(
                ck::Sequence<0>{}, ck::Sequence<1>{}, ck::Sequence<2>{}, ck::Sequence<3>{}));

        const auto in_n_y_ho_x_wo_c_grid_desc = ck::transform_tensor_descriptor(
            in_n_hip_wip_c_grid_desc,
            ck::make_tuple(ck::make_pass_through_transform(n),
                           ck::make_embed_transform(ck::make_tuple(fy, ho), ck::make_tuple(dy, sy)),
                           ck::make_embed_transform(ck::make_tuple(fx, wo), ck::make_tuple(dx, sx)),
                           ck::make_pass_through_transform(c)),
            ck::make_tuple(
                ck::Sequence<0>{}, ck::Sequence<1>{}, ck::Sequence<2>{}, ck::Sequence<3>{}),
            ck::make_tuple(
                ck::Sequence<0>{}, ck::Sequence<1, 2>{}, ck::Sequence<3, 4>{}, ck::Sequence<5>{}));

        const auto in_gemm_m_k_grid_desc = ck::transform_tensor_descriptor(
            in_n_y_ho_x_wo_c_grid_desc,
            ck::make_tuple(ck::make_merge_transform(ck::make_tuple(n, ho, wo)),
                           ck::make_merge_transform(ck::make_tuple(fy, fx, c))),
            ck::make_tuple(ck::Sequence<0, 2, 4>{}, ck::Sequence<1, 3, 5>{}),
            ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

        if constexpr(IsInputPadded)
        {
            const auto gemm_m_raw = n * ho * wo;
            const auto gemm_m_pad = ck::math::integer_least_multiple(gemm_m_raw, MC) - gemm_m_raw;
            const auto gemm_k_raw = c * fy * fx;
            const auto gemm_k_pad = ck::math::integer_least_multiple(gemm_k_raw, KC) - gemm_k_raw;

            const auto in_gemm_pm_pk_grid_desc = ck::transform_tensor_descriptor(
                in_gemm_m_k_grid_desc,
                ck::make_tuple(ck::make_right_pad_transform(gemm_m_raw, gemm_m_pad),
                               ck::make_right_pad_transform(gemm_k_raw, gemm_k_pad)),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}));

            if constexpr(IsInputCBlockTranspose)
            {
                constexpr auto I0             = ck::Number<0>{};
                constexpr auto I1             = ck::Number<1>{};
                const auto in_gemm_pm0_pk_pm1 = ck::transform_tensor_descriptor(
                    in_gemm_pm_pk_grid_desc,
                    ck::make_tuple(
                        ck::make_unmerge_transform(ck::make_tuple(
                            in_gemm_pm_pk_grid_desc.GetLength(I0) / CBlockMVector, CBlockMVector)),
                        ck::make_pass_through_transform(in_gemm_pm_pk_grid_desc.GetLength(I1))),
                    ck::make_tuple(ck::Sequence<0>{}, ck::Sequence<1>{}),
                    ck::make_tuple(ck::Sequence<0, 2>{}, ck::Sequence<1>{}));
                return in_gemm_pm0_pk_pm1;
            }
            else
                return in_gemm_pm_pk_grid_desc;
        }
        else
        {
            return in_gemm_m_k_grid_desc;
        }
    }();

    const auto input_cblock_desc = [&]() {
        if constexpr(IsInputCBlockTranspose)
        {
            const auto in_cblock_m_k_m8 = ck::make_naive_tensor_descriptor_packed(
                ck::make_tuple(MC / CBlockMVector, KC, CBlockMVector));
            return in_cblock_m_k_m8;
        }
        else
        {
            return ck::make_naive_tensor_descriptor_packed(ck::make_tuple(MC, KC));
        }
    }();

    constexpr auto get_dim_access_order = []() {
        if constexpr(IsInputCBlockTranspose)
            return ck::Sequence<1, 0, 2>{};
        else
            return ck::Sequence<0, 1>{};
    };

    constexpr auto get_slice_length = []() {
        if constexpr(IsInputCBlockTranspose)
            return ck::Sequence<MC / CBlockMVector, KC, CBlockMVector>{};
        else
            return ck::Sequence<MC, KC>{};
    };

    using threadwise_transfer_t = ck::cpu::ThreadwiseTensorSliceTransferAvx2<
        AType,                              // SrcData
        AType,                              // DstData
        decltype(input_desc),               // SrcDesc
        decltype(input_cblock_desc),        // DstDesc
        PassThrough,                        // ElementwiseOperation
        decltype(get_slice_length()),       // SliceLengths
        decltype(get_dim_access_order()),   // DimAccessOrder
        1,                                  // VectorDim
        1,                                  // ScalarPerVector
        ck::InMemoryDataOperationEnum::Set, // InMemoryDataOperationEnum
        false,                              // SrcResetCoordinateAfterRun
        true                                // DstResetCoordinateAfterRun
        >;

    static constexpr ck::index_t nDim =
        ck::remove_reference_t<decltype(input_desc)>::GetNumOfDimension();

    auto threadwise_transfer = threadwise_transfer_t{input_desc,
                                                     ck::make_zero_multi_index<nDim>(),
                                                     input_cblock_desc,
                                                     ck::make_zero_multi_index<nDim>(),
                                                     PassThrough{}};

    auto input_buf = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
        static_cast<AType*>(input_mem.mpDeviceBuf), input_mem.mMemSize / sizeof(AType));

    auto input_cblock = ck::cpu::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(
        static_cast<AType*>(input_cblock_mem.mpDeviceBuf),
        input_cblock_mem.mMemSize / sizeof(AType));

    constexpr auto fwd_move_step = []() {
        if constexpr(IsInputCBlockTranspose)
            return ck::make_multi_index(0, KC, 0); // m/8 * k * 8
        else
            return ck::make_multi_index(0, KC);
    };

    threadwise_transfer.RunGeneric(input_desc, input_buf, input_cblock_desc, input_cblock);

    printf("----------------------\n");

    threadwise_transfer.MoveSrcSliceWindow(input_desc, fwd_move_step());

    // threadwise_transfer.RunGeneric(input_desc,  input_buf , input_cblock_desc, input_cblock);

    dump_memory(reinterpret_cast<uint32_t*>(input_mem.mpDeviceBuf),
                input_mem.mMemSize / sizeof(AType));
    std::cout << "======================" << std::endl;
    dump_memory(reinterpret_cast<uint32_t*>(input_cblock_mem.mpDeviceBuf),
                input_cblock_mem.mMemSize / sizeof(AType));
}
