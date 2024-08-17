// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "permute.hpp"
#include "ck_tile/host.hpp"

#include <array>
#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace detail {
template <int bytes>
struct to_integer_type;

template <>
struct to_integer_type<4>
{
    using type = int32_t;
};
template <>
struct to_integer_type<2>
{
    using type = int16_t;
};
template <>
struct to_integer_type<1>
{
    using type = int8_t;
};
} // namespace detail

template <int bytes>
using to_integer_type = typename detail::to_integer_type<bytes>::type;

// host API (shoule come from codegen)
float permute(permute_traits t, permute_args a, const ck_tile::stream_config& s)
{
    if(t.data_type.compare("fp8") == 0)
    {
        using DataType        = ck_tile::fp8_t;
        using PipelineProblem = ck_tile::GenericPermuteProblem<DataType>;
        using Kernel          = ck_tile::GenericPermute<PipelineProblem>;

        auto kargs = Kernel::MakeKargs(a);

        const dim3 grids      = Kernel::GridSize(a);
        constexpr dim3 blocks = Kernel::BlockSize();

        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, 1>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    }
    else if(t.data_type.compare("fp16") == 0)
    {
        using DataType        = ck_tile::half_t;
        using PipelineProblem = ck_tile::GenericPermuteProblem<DataType>;
        using Kernel          = ck_tile::GenericPermute<PipelineProblem>;

        auto kargs = Kernel::MakeKargs(a);

        const dim3 grids      = Kernel::GridSize(a);
        constexpr dim3 blocks = Kernel::BlockSize();

        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, 1>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    }
    else if(t.data_type.compare("fp32") == 0)
    {
        using DataType        = float;
        using PipelineProblem = ck_tile::GenericPermuteProblem<DataType>;
        using Kernel          = ck_tile::GenericPermute<PipelineProblem>;

        auto kargs = Kernel::MakeKargs(a);

        const dim3 grids      = Kernel::GridSize(a);
        constexpr dim3 blocks = Kernel::BlockSize();

        float ave_time = ck_tile::launch_kernel(
            s, ck_tile::make_kernel<blocks.x, 1>(Kernel{}, grids, blocks, 0, kargs));

        return ave_time;
    }

    return 0;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    using size_type = typename std::vector<T>::size_type;

    os << "[";
    for(size_type idx = 0; idx < v.size(); ++idx)
    {
        if(0 < idx)
        {
            os << ", ";
        }
        os << v[idx];
    }
    return os << "]";
}

auto create_args(int argc, char* argv[])
{
    ck_tile::ArgParser arg_parser;
    arg_parser.insert("v", "1", "weather do CPU validation or not")
        .insert("prec", "fp16", "data type. fp8/fp16/fp32 (representing 8/16/32 bit data)")
        .insert("shape", "2,3,4", "the shape of the input tensor")
        .insert("perm", "2,1,0", "permute perm")
        .insert("kname", "0", "t to 1 will print kernel name")
        .insert("seed",
                "11939",
                "random seed used for initializing input tensors. 0 for "
                "non-deterministic seed")
        .insert("warmup", "5", "number of iterations before benchmark the kernel")
        .insert("repeat", "20", "number of iterations to benchmark the kernel");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

// different threshold for different dtype
template <typename DataType>
auto get_elimit(std::string /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::bf16_t>(std::string /*init_method*/)
{
    double rtol = 1e-2;
    double atol = 1e-2;
    return ck_tile::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck_tile::fp8_t>(std::string init_method)
{
    if(init_method == "ui" || init_method == "ni")
    {
        unsigned max_rounding_point_distance = 0;
        double atol                          = 2e-3;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
    else
    {
        unsigned max_rounding_point_distance = 1;
        double atol                          = 0.0625;
        return ck_tile::make_tuple(max_rounding_point_distance, atol);
    }
}

// "1,2,3,4" -> vector{1,2,3,4}
std::vector<ck_tile::index_t> decode_vec(std::string q_val)
{
#define _S2I_(str_) static_cast<ck_tile::index_t>(std::atoi((str_).c_str()))
    std::string::size_type pos = 0;
    std::vector<ck_tile::index_t> v;
    while(true)
    {
        auto found = q_val.find(',', pos);
        ck_tile::index_t n =
            _S2I_(q_val.substr(pos, found == std::string::npos ? found : found - pos));
        v.push_back(n);
        if(found == std::string::npos)
        {
            break;
        }
        pos = found + 1;
    }
    return v;
#undef _S2I_
}

template <typename DataType>
bool run(const ck_tile::ArgParser& arg_parser)
{
    std::string data_type = arg_parser.get_str("prec");
    int do_validation     = arg_parser.get_int("v");

    auto x_shape      = decode_vec(arg_parser.get_str("shape"));
    auto perm         = decode_vec(arg_parser.get_str("perm"));
    int stream_warmup = arg_parser.get_int("warmup");
    int stream_repeat = arg_parser.get_int("repeat");
    bool kname        = arg_parser.get_bool("kname");
    int seed          = arg_parser.get_int("seed");

    assert(shape.size() == perm.size());
    ck_tile::index_t rank = perm.size();
    if(rank > ck_tile::GenericPermuteHostArgs::kMaxRanks)
    {
        printf("rank %d permute is not support yet\n", rank);
        return false;
    }

    ck_tile::HostTensor<DataType> x(x_shape);
    ck_tile::FillUniformDistributionIntegerValue<DataType>{-15, 15, seed}(x);

    std::vector<ck_tile::index_t> y_shape = [&]() {
        std::vector<ck_tile::index_t> tmp(rank, 0);
        // std::cout << "@@@@" << tmp << std::endl;
        for(int i = 0; i < static_cast<int>(rank); i++)
        {
            // std::cout << "  i:" << i << ", perm:" << perm[i] << ", rak:" <<
            // static_cast<int>(rank)
            // << std::endl;
            tmp[i] = x_shape[perm[i]];
        }
        // std::cout << "@@@" << tmp << std::endl;
        return tmp;
    }();

    ck_tile::HostTensor<DataType> y(y_shape);

    ck_tile::DeviceMem x_buf(x.get_element_space_size_in_bytes());
    ck_tile::DeviceMem y_buf(y.get_element_space_size_in_bytes());

    x_buf.ToDevice(x.data());

    permute_args args;
    args.p_src = x_buf.GetDeviceBuffer();
    args.p_dst = y_buf.GetDeviceBuffer();
    args.rank  = rank;
    std::copy(x_shape.begin(), x_shape.end(), args.shape);
    std::copy(perm.begin(), perm.end(), args.perm);

    permute_traits trait;
    trait.data_type = data_type;

    std::cout << "[" << data_type << "] shape:" << x_shape << "->" << y_shape
              << ", permute:" << perm << std::flush;

    ck_tile::stream_config stream_config{nullptr,
                                         true,
                                         /* log_level = */ (kname ? 1 : 0),
                                         stream_warmup,
                                         stream_repeat};

    float ave_time = permute(trait, args, stream_config);
    std::cout << ", time:" << ave_time << "ms" << std::flush;

    bool pass = true;
    if(do_validation)
    {
        reference_permute(x, y, perm);
#if 0
        if constexpr (std::is_same_v<float, DataType>){
            // using itype = to_integer_type<sizeof(DataType)>;
            fflush(stdout);
            for(int zz = 0; zz < static_cast<int>(x.get_element_size()); zz++   ) {
                printf("%3.0f ", x.mData[zz]);
            }
            printf("->\n");
            for(int zz = 0; zz < static_cast<int>(x.get_element_size()); zz++   ) {
                printf("%3.0f ", y.mData[zz]);
            }
            fflush(stdout);
        }
#endif
        ck_tile::HostTensor<DataType> y_dev(y.get_lengths());

        y_buf.FromDevice(y_dev.data());

        pass = std::equal(
            y_dev.begin(), y_dev.end(), y.begin(), [&](const DataType& d, const DataType& h) {
                using itype = to_integer_type<sizeof(DataType)>;
                itype i_d   = ck_tile::bit_cast<itype>(d);
                itype i_h   = ck_tile::bit_cast<itype>(h);
                return i_d == i_h;
            });
        std::cout << ", valid:" << (pass ? "y" : "n") << std::flush;
    }

    std::cout << std::endl;

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp8")
    {
        return run<ck_tile::fp8_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "fp16")
    {
        return run<ck_tile::half_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "fp32")
    {
        return run<float>(arg_parser) ? 0 : -2;
    }

    return -3;
}
