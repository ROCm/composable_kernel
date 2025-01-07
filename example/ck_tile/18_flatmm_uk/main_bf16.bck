#include <algorithm>
#include <cstring>
#include <unordered_set>
#include <vector>
#include <set>

#include "ck_tile/host.hpp"
#include "flatmm_uk.hpp"

// different threshold for different dtype
template <typename DataType>
auto get_elimit()
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>()
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename AElementOp   = ck_tile::identity,
          typename BElementOp   = ck_tile::identity,
          typename ACCElementOp = ck_tile::identity>
CK_TILE_HOST void my_reference_gemm(const ck_tile::HostTensor<ADataType>& a_m_k,
                                    const ck_tile::HostTensor<BDataType>& b_k_n,
                                    ck_tile::HostTensor<CDataType>& c_m_n,
                                    float t,
                                    const AElementOp& a_element_op     = {},
                                    const BElementOp& b_element_op     = {},
                                    const ACCElementOp& acc_element_op = {})
{
    const std::size_t M = a_m_k.get_length(0);
    const std::size_t N = b_k_n.get_length(0);
    const std::size_t K = a_m_k.get_length(1);
    printf("[REF] M = %zu, N = %zu, K = %zu\n", M, N, K);

    auto cal_tflops = [&](auto ms) {
        double flop_gemm = 2.0 * M * N * K;
        return (flop_gemm) / (static_cast<double>(ms) * 1e-3) / 1e12;
    };

    auto cal_tbps = [&](auto ms) {
        double a_bytes = static_cast<double>(M) * K * sizeof(ADataType);
        double b_bytes = static_cast<double>(N) * K * sizeof(BDataType);
        double o_bytes = static_cast<double>(M) * N * sizeof(CDataType);

        return (a_bytes + b_bytes + o_bytes) / (static_cast<double>(ms) * 1e-3) / 1e12;
    };

    std::cout << ", " << t * 1.E3 << " us, " << cal_tflops(t) << " tflops, " << cal_tbps(t)
              << " TB/s" << std::endl
              << std::flush;

    auto f_mn = [&](auto m, auto n) {
        AccDataType v_acc = 0;

        for(std::size_t k = 0; k < K; ++k)
        {
            ADataType v_a = a_element_op(a_m_k(m, k));
            BDataType v_b = b_element_op(b_k_n(n, k));

            v_acc +=
                ck_tile::type_convert<AccDataType>(v_a) * ck_tile::type_convert<AccDataType>(v_b);
        }

        c_m_n(m, n) = ck_tile::type_convert<CDataType>(acc_element_op(v_acc));
    };

    ck_tile::make_ParallelTensorFunctor(f_mn, M, N)(std::thread::hardware_concurrency());
}

// mfma_type, 0:32x32, 1:16x16
// TODO: padding?
template <typename T>
auto shuffle_moe_weight(const ck_tile::HostTensor<T>& t, std::string mfma_dtype, int mfma_type = 0)
{
    assert(t.get_lengths().size() == 3);
    int b_ = t.get_lengths()[0];
    int n_ = t.get_lengths()[1];
    int k_ = t.get_lengths()[2];
    if((mfma_dtype == "bf16" || mfma_dtype == "fp16") && mfma_type == 0)
    {
        ck_tile::HostTensor<T> t_view({b_, n_ / 32, 32, k_ / 16, 2, 8});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 1, 3, 4, 2, 5});
    }
    else if((mfma_dtype == "bf16" || mfma_dtype == "fp16") && mfma_type == 1)
    {
        ck_tile::HostTensor<T> t_view({b_, n_ / 16, 16, k_ / 32, 4, 8});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 1, 3, 4, 2, 5});
    }
    else if((mfma_dtype == "int8" || mfma_dtype == "fp8") && mfma_type == 0)
    {
        ck_tile::HostTensor<T> t_view({b_, n_ / 32, 32, k_ / 32, 2, 16});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 1, 3, 4, 2, 5});
    }
    else if((mfma_dtype == "int8" || mfma_dtype == "fp8") && mfma_type == 1)
    {
        ck_tile::HostTensor<T> t_view({b_, n_ / 16, 16, k_ / 64, 4, 16});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 1, 3, 4, 2, 5});
    }
    return t;
}
template <typename T>
auto shuffle_weight(const ck_tile::HostTensor<T>& t, std::string mfma_dtype, int mfma_type = 0)
{
    assert(t.get_lengths().size() == 2);
    int n_ = t.get_lengths()[0];
    int k_ = t.get_lengths()[1];
    if((mfma_dtype == "bf16" || mfma_dtype == "fp16") && mfma_type == 0)
    {
        ck_tile::HostTensor<T> t_view({n_ / 32, 32, k_ / 16, 2, 8});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else if((mfma_dtype == "bf16" || mfma_dtype == "fp16") && mfma_type == 1)
    {
        ck_tile::HostTensor<T> t_view({n_ / 16, 16, k_ / 32, 4, 8});
        printf("[FF] permute: n_ = %d, k_ = %d, n_/16 = %d, k_/32 = %d\n", n_, k_, n_ / 16, k_ / 32);
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else if((mfma_dtype == "int8" || mfma_dtype == "fp8") && mfma_type == 0)
    {
        ck_tile::HostTensor<T> t_view({n_ / 32, 32, k_ / 32, 2, 16});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    else if((mfma_dtype == "int8" || mfma_dtype == "fp8") && mfma_type == 1)
    {
        ck_tile::HostTensor<T> t_view({n_ / 16, 16, k_ / 64, 4, 16});
        std::copy(t.begin(), t.end(), t_view.begin());
        return ck_tile::reference_permute(t_view, {0, 2, 3, 1, 4});
    }
    return t;
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("m", "64", "num of m")
        .insert("n", "1024", "num of n")
        .insert("k", "8192", "num of k")
        .insert("t", "64", "num input tokens")
        .insert("e", "8", "num of experts")
        .insert("tk", "1", "topk")
        .insert("h", "4096", "hidden_size of this model")
        .insert("i", "4096", "intermediate_size between 2 gemms of FFN")
        .insert("stride", "-1", "stride per row, if -1 then equal to hidden_size")
        .insert("bm", "32", "blocking factor for sorted tokens")
        .insert("tp", "8", "tensor parallel size")
        .insert("v", "1", "cpu validation or not")
        .insert("kname", "1", "print kernel name or not")
        .insert("prec_i", "bf16", "input precision")
        .insert("prec_w", "bf16", "weight precision")
        .insert("prec_o", "bf16", "output precision")
        .insert("prec_st", "auto", "token scale data type. auto will set to fp32")
        .insert("prec_sw", "auto", "weight scale data type. auto will set to fp32")
        .insert("prec_sq", "auto", "(dynamic) smooth quant data type. auto will set to fp32")
        .insert("prec_kw", "auto", "topk-weight data type. auto will set to fp32")
        .insert("fquant", "0", "fused-quant, 0:no, 1:smooth-dynamic-quant, 2:dynamic-quant")
        .insert(
            "gate_only", "1", "w0(gate/up) style, 0:gate+up will double interm size, 1:only gate")
        .insert("api", "0", "benchmark api set: 0:fused-moe(moe-gemm+moe-sorting), 1:moe-gemm")
        .insert("balance",
                "0",
                "if set to 1, will try balance the expert in topk-ids(convenient for testing)")
        .insert("init",
                "2",
                "init method. 0:random stepped float(fast). 1: random uniform, 2:rand normalized"
                "normalized(slow)")
        .insert("seed", "11939", "seed used to do random")
        .insert("warmup", "1", "cold iter")
        .insert("repeat", "4", "hot iter");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// I:input-type, W:weight-type, O:output-type, ST:toke-scale-tpye, SW:weight-scale-type,
// SQ:smooth-quant-type, KW:topk-weight-type
template <typename I, typename W, typename O, typename ST, typename SW, typename SQ, typename KW>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");
    printf("[FF] M = %d, N = %d, K = %d\n", M, N, K);

    ck_tile::index_t experts     = arg_parser.get_int("e");
    ck_tile::index_t topk        = arg_parser.get_int("tk");
    ck_tile::index_t stride      = arg_parser.get_int("stride");
    ck_tile::index_t block_m     = arg_parser.get_int("bm");
    std::string prec_i  = arg_parser.get_str("prec_i");
    std::string prec_w  = arg_parser.get_str("prec_w");
    std::string prec_o  = arg_parser.get_str("prec_o");
    std::string prec_st = arg_parser.get_str("prec_st");
    std::string prec_sw = arg_parser.get_str("prec_sw");
    std::string prec_sq = arg_parser.get_str("prec_sq");
    std::string prec_kw = arg_parser.get_str("prec_kw");
    prec_st             = (prec_st == "auto") ? "fp32" : prec_st;
    prec_sw             = (prec_sw == "auto") ? "fp32" : prec_sw;
    prec_sq             = (prec_sq == "auto") ? "fp32" : prec_sq;
    prec_kw             = (prec_kw == "auto") ? "fp32" : prec_kw;
    int kname           = arg_parser.get_int("kname");
    int do_validation   = arg_parser.get_int("v");
    int warmup          = arg_parser.get_int("warmup");
    int repeat          = arg_parser.get_int("repeat");
    int fused_quant     = arg_parser.get_int("fquant");
    int gate_only       = arg_parser.get_int("gate_only");
    int init            = arg_parser.get_int("init");
    uint32_t seed       = arg_parser.get_uint32("seed");

    using TypeConfig  = FlatmmUkTypeConfig<I, W, O, ST, SW, SQ, KW>;
    using ADataType   = typename TypeConfig::ADataType;
    using BDataType   = ADataType;
    using AccDataType = typename TypeConfig::AccDataType;
    using CDataType   = AccDataType;
    using DDataType   = AccDataType;

    // host verify
    ck_tile::HostTensor<ADataType> a_host({M, K});
    ck_tile::HostTensor<BDataType> b_host({N, K});
    ck_tile::HostTensor<CDataType> c_host({M, N});
    ck_tile::HostTensor<DDataType> d_host({M, N});

    ck_tile::HostTensor<int> dbg_int({M * N, K});
    ck_tile::HostTensor<float> dbg_fp32({M * N, K});
    ck_tile::HostTensor<ck_tile::bf16_t> dbg_bf16({M * N, K});

    if(init == 0)
    {
        ck_tile::FillStepRange<ADataType>{-.5f, .5f, 0.01f}(a_host);
        ck_tile::FillStepRange<BDataType>{-.5f, .5f, 0.01f}(b_host);
    }
    else if(init == 1)
    {
        ck_tile::FillUniformDistribution<ADataType>{-.5f, .5f, seed, true}(a_host);
        ck_tile::FillUniformDistribution<BDataType>{-.5f, .5f, seed, true}(b_host);
    }
    else if(init == 2)
    {
        ck_tile::FillNormalDistribution<ADataType>{0.f, 1.f, seed, true}(a_host);
        ck_tile::FillNormalDistribution<BDataType>{0.f, 1.f, seed, true}(b_host);
    }
    /*
    // a_host
    {
        int X = static_cast<int>(K);
        int Y = static_cast<int>(M);

        for(int y = 0; y < Y; y++)
        {
            for(int x = 0; x < X; x++)
            {
                int idx = X * y + x;
                a_host.mData[idx] = ck_tile::type_convert<ADataType>(x * 1.0f);
                //b_host.mData[idx] = ck_tile::type_convert<GDataType>(y * 1.0f);
                //b_host.mData[idx] = ck_tile::type_convert<GDataType>(y*1.f + x * 0.0001f);
            }
        }
    }
    // b_host
    {
        int X = static_cast<int>(K);
        int Y = static_cast<int>(N);

        for(int y = 0; y < Y; y++)
        {
            for(int x = 0; x < X; x++)
            {
                int idx = X * y + x;
                b_host.mData[idx] = ck_tile::type_convert<GDataType>(idx * 1.0f);
                //b_host.mData[idx] = ck_tile::type_convert<GDataType>(y * 1.0f);
                //b_host.mData[idx] = ck_tile::type_convert<GDataType>(y*1.f + x * 0.0001f);
            }
        }
    }*/

    // permute weight
    ck_tile::HostTensor<BDataType> b_perm_host = shuffle_weight(b_host, prec_w, 1);

    ck_tile::DeviceMem a_buf(a_host);
    ck_tile::DeviceMem b_buf(b_perm_host); // b_host -> b_perm_host
    ck_tile::DeviceMem c_buf(c_host);
    ck_tile::DeviceMem d_buf(d_host);
    ck_tile::DeviceMem dbg_int_buf(dbg_int);
    ck_tile::DeviceMem dbg_bf16_buf(dbg_bf16);
    ck_tile::DeviceMem dbg_fp32_buf(dbg_fp32);

    flatmm_uk_traits traits{prec_i,
                            prec_w,
                            prec_o,
                            prec_st,
                            prec_sw,
                            prec_sq,
                            prec_kw,
                            block_m,
                            gate_only,
                            fused_quant};
    printf("[FF] --- run(): <flatmm_uk_traits> ---\n");
    printf("[FF] traits.prec_i = %s\n", traits.prec_i.c_str());
    printf("[FF] traits.prec_w = %s\n", traits.prec_w.c_str());
    printf("[FF] traits.prec_o = %s\n", traits.prec_o.c_str());
    printf("[FF] traits.prec_st = %s\n", traits.prec_st.c_str());
    printf("[FF] traits.prec_sw = %s\n", traits.prec_sw.c_str());
    printf("[FF] traits.prec_sq = %s\n", traits.prec_sq.c_str());
    printf("[FF] traits.prec_kw = %s\n", traits.prec_kw.c_str());
    printf("[FF] traits.block_m = %d\n", traits.block_m);
    printf("[FF] traits.gate_only = %d\n", traits.gate_only);
    printf("[FF] traits.fused_quant = %d\n", traits.fused_quant);

    flatmm_uk_args args{a_buf.GetDeviceBuffer(),
                        b_buf.GetDeviceBuffer(),
                        c_buf.GetDeviceBuffer(),
                        d_buf.GetDeviceBuffer(),
                        dbg_int_buf.GetDeviceBuffer(),
                        dbg_bf16_buf.GetDeviceBuffer(),
                        dbg_fp32_buf.GetDeviceBuffer(),
                        block_m,
                        K,
                        N,
                        M,
                        experts,
                        topk,
                        stride};
    printf("[FF] --- run(): <flatmm_uk_args> ---\n");
    printf("[FF] args.block_m = %d\n", args.block_m);
    printf("[FF] args.hidden_size = %d\n", args.hidden_size);
    printf("[FF] args.intermediate_size = %d\n", args.intermediate_size);
    printf("[FF] args.num_tokens = %d\n", args.num_tokens);   // 1
    printf("[FF] args.topk = %d\n", args.topk);               // 0
    printf("[FF] args.num_experts = %d\n", args.num_experts); // 0
    printf("[FF] args.stride_token = %d\n", args.stride_token);

    float ave_time = flatmm_uk(
        traits, args, ck_tile::stream_config{nullptr, true, kname ? 1 : 0, warmup, repeat});

    if(ave_time < 0)
    {
        std::cout << " not supported!" << std::endl << std::flush;
        return false;
    }

    bool pass = true;

    if(do_validation)
    {
        auto d_dev = d_buf.ToHost<float>();
        std::cout << std::endl << " =================== " << std::endl;
        d_host.SetZero();
        my_reference_gemm<ADataType, BDataType, CDataType, DDataType>(
            a_host, b_host, d_host, ave_time);
        pass = ck_tile::check_err(d_dev, d_host);
        std::cout << "The CPU veification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

#if 0
    int GridDimX  = 2;
    int GridDimY  = 1;
    int BlockDimX = 64;
    int BlockDimY = 4;
    int BlockSize = BlockDimX * BlockDimY;
    // dbg_int
    {
        auto dbg_int_dev = dbg_int_buf.ToHost<int>();
        std::ofstream file("ff_dbg_int.txt");
        file << " [dbg_int]: Grid = [" << GridDimX << ", " << GridDimY << "], Block = " << BlockSize
             << std::endl;

        for(int bidy = 0; bidy < GridDimY; bidy++)
        {
            for(int bidx = 0; bidx < GridDimX; bidx++)
            {
                file << "\n ========== block : [" << bidx << ", " << bidy << "] ==========";
                for(int tid = 0; tid < BlockSize; tid++)
                {
                    int gid = (BlockSize * GridDimX) * bidy + BlockSize * bidx + tid;
                    if(tid % 64 == 0)
                    {
                        file << "\n [" << tid << " : " << tid + 63 << "]: ";
                    }
                    file << ck_tile::type_convert<int>(dbg_int_dev.mData[gid]) << ", ";
                }
            }
        }

        file.close();
    }
    // dbg_bf16 ---> kernel
    {
        auto dbg_bf16_dev = dbg_bf16_buf.ToHost<BDataType>();
        std::ofstream file("ff_dbg_bf16_kernel.txt");
        file << " [dbg_bf16]: Grid = [" << GridDimX << ", " << GridDimY
             << "], Block = " << BlockSize << std::endl;

        for(int bidy = 0; bidy < GridDimY; bidy++)
        {
            for(int bidx = 0; bidx < GridDimX; bidx++)
            {
                file << "\n ========== block : [" << bidx << ", " << bidy << "] ==========";
                for(int tid = 0; tid < BlockSize; tid++)
                {
                    int gid = (BlockSize * bidx) * bidy + BlockSize * bidx + tid;

                    file << "\n [" << tid << "]: ";
                    for(int i = 0; i < 64; i++) // multi output per thread
                        file << ck_tile::type_convert<float>(dbg_bf16_dev.mData[gid * 64 + i])
                             << ", ";
                }
            }
        }

        file.close();
    }
    // dbg_bf16
    {
        auto dbg_bf16_dev = dbg_bf16_buf.ToHost<BDataType>();
        std::ofstream file("ff_dbg_bf16.txt");
        int X = static_cast<int>(N);
        int Y = static_cast<int>(M);
        file << " [dbg_bf16]: Row = " << Y << ", Col = " << X << std::endl;

        for(int m = 0; m < Y; m++)
        {
            file << "\n ========== row : [" << m << " / " << Y << "] ==========";
            for(int n = 0; n < X; n++)
            {
                if(n % 64 == 0)
                {
                    file << "\n [" << n << " : " << n + 63 << "]: ";
                }
                int idx = X * m + n;
                file << ck_tile::type_convert<float>(dbg_bf16_dev.mData[idx]) << ", ";
            }
        }

        file.close();
    }
    // dbg_fp32 ---> kernel
    {
        auto dbg_fp32_dev = dbg_fp32_buf.ToHost<float>();
        std::ofstream file("ff_dbg_fp32_kernel.txt");
        file << " [dbg_fp32]: Grid = [" << GridDimX << ", " << GridDimY
             << "], Block = " << BlockSize << std::endl;

        for(int bidy = 0; bidy < GridDimY; bidy++)
        {
            for(int bidx = 0; bidx < GridDimX; bidx++)
            {
                file << "\n ========== block : [" << bidx << ", " << bidy << "] ==========";
                for(int tid = 0; tid < BlockSize; tid++)
                {
                    int gid = (BlockSize * bidx) * bidy + BlockSize * bidx + tid;

                    file << "\n [" << tid << "]: ";
                    for(int i = 0; i < 64; i++) // multi output per thread
                        file << ck_tile::type_convert<float>(dbg_fp32_dev.mData[gid * 64 + i])
                             << ", ";

                    // if(tid % 64 == 0) // one output per thread
                    //     file << "\n [" << tid << " : " << tid + 63 << "]: ";
                    // file << ck_tile::type_convert<float>(dbg_bf16.mData[gid]) << ", ";
                }
            }
        }

        file.close();
    }
    // dbg_fp32
    {
        auto dbg_fp32_dev = dbg_fp32_buf.ToHost<float>();
        std::ofstream file("ff_dbg_fp32.txt");
        int X = static_cast<int>(N);
        int Y = static_cast<int>(M);
        file << " [dbg_fp32]: Row = " << Y << ", Col = " << X << std::endl;

        for(int m = 0; m < Y; m++)
        {
            file << "\n ========== row : [" << m << " / " << Y << "] ==========";
            for(int n = 0; n < X; n++)
            {
                if(n % 64 == 0)
                {
                    file << "\n [" << n << " : " << n + 63 << "]: ";
                }
                int idx = X * m + n;
                file << ck_tile::type_convert<float>(dbg_fp32_dev.mData[idx]) << ", ";
            }
        }

        file.close();
    }
    // a_host
    {
        std::ofstream file("ff_a_host.txt");
        int X = static_cast<int>(K);
        int Y = static_cast<int>(M);
        file << " [a_host]: Row = " << Y << ", Col = " << X << std::endl;

        for(int y = 0; y < Y; y++)
        {
            file << "\n ========== row : [" << y << " / " << Y << "] ==========";
            for(int x = 0; x < X; x++)
            {
                int idx = X * y + x;
                if(idx % 16 == 0)
                {
                    file << "\n [" << x << " : " << x + 15 << " ]: ";
                }

                file << ck_tile::type_convert<float>(a_host.mData[idx]) << ", ";
            }
        }

        file.close();
    }
    // b_host
    {
        std::ofstream file("ff_b_host.txt");
        int X = static_cast<int>(K);
        int Y = static_cast<int>(N);
        file << " [b_host]: Row = " << Y << ", Col = " << X << std::endl;

        for(int y = 0; y < Y; y++)
        {
            file << "\n ========== row : [" << y << " / " << Y << "] ==========";
            for(int x = 0; x < X; x++)
            {
                int idx = X * y + x;
                if(idx % 16 == 0)
                {
                    file << "\n [" << x << " : " << x + 15 << " ]: ";
                }

                file << ck_tile::type_convert<float>(b_host.mData[idx]) << ", ";
            }
        }

        file.close();
    }
    // permute_b
    {
        std::ofstream file("ff_b_perm_host.txt");
        int X = static_cast<int>(K);
        int Y = static_cast<int>(N);
        file << " [b_perm_host]: Row = " << Y << ", Col = " << X << std::endl;

        for(int y = 0; y < Y; y++)
        {
            file << "\n ========== row : [" << y << " / " << Y << "] ==========";
            for(int x = 0; x < X; x++)
            {
                int idx = X * y + x;
                if(idx % 16 == 0)
                {
                    file << "\n [" << x << " : " << x + 15 << " ]: ";
                }

                file << ck_tile::type_convert<float>(b_perm_host.mData[idx]) << ", ";
            }
        }

        file.close();
    }
    // d_dev ---> kernel
    {
        auto d_dev = d_buf.ToHost<float>();
        std::ofstream file("ff_d_dev_kernel.txt");
        file << " [d_dev]: Grid = [" << GridDimX << ", " << GridDimY << "], Block = " << BlockSize
             << std::endl;

        for(int bidy = 0; bidy < GridDimY; bidy++)
        {
            for(int bidx = 0; bidx < GridDimX; bidx++)
            {
                file << "\n ========== block : [" << bidx << ", " << bidy << "] ==========";
                for(int tid = 0; tid < BlockSize; tid++)
                {
                    int gid = (BlockSize * bidx) * bidy + BlockSize * bidx + tid;

                    file << "\n [" << tid << "]: ";
                    for(int i = 0; i < 64; i++) // multi output per thread
                        file << ck_tile::type_convert<float>(d_dev.mData[gid * 64 + i]) << ", ";
                }
            }
        }

        file.close();
    }
    // d_dev
    {
        auto d_dev = d_buf.ToHost<float>();
        std::ofstream file("ff_d_dev.txt");
        int X = static_cast<int>(N);
        int Y = static_cast<int>(M);
        file << " [d_dev]: Row = " << Y << ", Col = " << X << std::endl;

        for(int y = 0; y < Y; y++)
        {
            file << "\n ========== row : [" << y << " / " << Y << "] ==========";
            for(int x = 0; x < X; x++)
            {
                if(x % 64 == 0)
                {
                    file << "\n [" << x << " : " << x + 63 << "]: ";
                }
                int idx = X * y + x;
                file << ck_tile::type_convert<float>(d_dev.mData[idx]) << ", ";
            }
        }

        file.close();
    }
    // d_host
    {
        std::ofstream file("ff_d_host.txt");
        int X = static_cast<int>(N);
        int Y = static_cast<int>(M);
        file << " [d_host]: Row = " << Y << ", Col = " << X << std::endl;

        for(int y = 0; y < Y; y++)
        {
            file << "\n ========== row : [" << y << " / " << Y << "] ==========";
            for(int x = 0; x < X; x++)
            {
                if(x % 64 == 0)
                {
                    file << "\n [" << x << " : " << x + 63 << "]: ";
                }
                int idx = X * y + x;
                file << ck_tile::type_convert<float>(d_host.mData[idx]) << ", ";
            }
        }

        file.close();
    }
#endif

    std::cout << std::flush << std::endl;
    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    std::string prec_i  = arg_parser.get_str("prec_i");
    std::string prec_w  = arg_parser.get_str("prec_w");
    std::string prec_o  = arg_parser.get_str("prec_o");
    std::string prec_st = arg_parser.get_str("prec_st");
    std::string prec_sw = arg_parser.get_str("prec_sw");
    std::string prec_sq = arg_parser.get_str("prec_sq");
    std::string prec_kw = arg_parser.get_str("prec_kw");
    prec_st             = (prec_st == "auto") ? "fp32" : prec_st;
    prec_sw             = (prec_sw == "auto") ? "fp32" : prec_sw;
    prec_sq             = (prec_sq == "auto") ? "fp32" : prec_sq;
    prec_kw             = (prec_kw == "auto") ? "fp32" : prec_kw;

    // no dynamic quant case
    if(prec_i == "bf16" && prec_w == "bf16" && prec_o == "bf16" && prec_kw == "fp32")
    {
        return run<ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t, float, float, float, float>(
                   arg_parser)
                   ? 0
                   : -2;
    }
    else if(prec_i == "fp16" && prec_w == "fp16" && prec_o == "fp16" && prec_kw == "fp32")
    {
        return run<ck_tile::fp16_t, ck_tile::fp16_t, ck_tile::fp16_t, float, float, float, float>(
                   arg_parser)
                   ? 0
                   : -2;
    }

    return -3;
}
