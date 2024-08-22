#include "matrix_core_swizzle.hpp"
#include "matrix_core_swizzle_kernel.hpp"

float matrix_core_swizzle(matrix_core_swizzle_traits t,
                          matrix_core_swizzle_args a,
                          const ck_tile::stream_config& s)
{
    if(t.data_type.compare("fp16") == 0)
    {
        if(t.inst.compare("32x32x8") == 0)
        {
            constexpr int BLOCK_SIZE             = 256;
            constexpr int NPerBlock              = 256;
            constexpr int KPerBlock              = 128;
            constexpr matrix_core_inst_enum Inst = matrix_core_inst_enum::MFMA_32x32x8_F16;
            if(t.permute.compare("0,1,4,2,5,3,6") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
            else if(t.permute.compare("0,1,2,4,5,3,6") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_n0_n1_k0_k1_n2_k2;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
            else if(t.permute.compare("0,1,3,4,2,5") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_nr_kr_kw_nw_kv;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
        }
        else if(t.inst.compare("16x16x16") == 0)
        {
            constexpr int BLOCK_SIZE             = 256;
            constexpr int NPerBlock              = 256;
            constexpr int KPerBlock              = 128;
            constexpr matrix_core_inst_enum Inst = matrix_core_inst_enum::MFMA_16x16x16_F16;
            if(t.permute.compare("0,1,4,2,5,3,6") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_n0_k0_n1_k1_n2_k2;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
            else if(t.permute.compare("0,1,2,4,5,3,6") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_n0_n1_k0_k1_n2_k2;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
            else if(t.permute.compare("0,1,3,4,2,5") == 0)
            {
                constexpr matrix_core_permute_style pstyle =
                    matrix_core_permute_style::permute_b_nr_kr_kw_nw_kv;
                using Kernel =
                    matrix_core_swizzle_kernel<BLOCK_SIZE, NPerBlock, KPerBlock, pstyle, Inst>;

                auto k         = Kernel(a);
                float ave_time = ck_tile::launch_kernel(s, k);

                return ave_time;
            }
        }
    }
    return -1;
}
