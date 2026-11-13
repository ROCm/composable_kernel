// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// bf16 MoE GEMM with a model dim that is not a multiple of the tile, using
// GemmSpecialization::NKPadding. MOE_STAGE=1 is the gate/up GEMM (model dim is K), MOE_STAGE=2
// the down GEMM (model dim is N). Each expert's B is zero-padded only to whole preshuffle tiles
// (N to NPerXdl, K to KLane * KPack) before preshuffling; A and E keep their logical sizes.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"

#ifndef MOE_STAGE
#define MOE_STAGE 1
#endif

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using BF16        = ck::bhalf_t;
using F32         = float;
using Row         = ck::tensor_layout::gemm::RowMajor;
using Col         = ck::tensor_layout::gemm::ColumnMajor;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using DsDataType  = ck::Tuple<F32, F32, F32>;
using DsLayout    = ck::Tuple<Row, Col, Row>;
using ck::DeviceMem;

struct TypeCast
{
    template <typename E, typename C, typename D0, typename D1, typename D2>
    __host__ __device__ constexpr void
    operator()(E& e, const C& c, const D0&, const D1&, const D2&) const
    {
        e = ck::type_convert<E>(c);
    }
};

static constexpr bool IsStage1 = MOE_STAGE == 1;
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::NKPadding;
static constexpr ck::index_t MPerBlock = IsStage1 ? 32 : 64;
static constexpr ck::index_t NPerXdl   = 16;
static constexpr ck::index_t KPack     = 16 / sizeof(BF16);
static constexpr ck::index_t KLane     = 64 / NPerXdl;

// clang-format off
using DeviceOpStage1 = ck::tensor_operation::device::DeviceMoeGemm<
    Row, Col, DsLayout, Row, BF16, BF16, DsDataType, BF16, F32, F32,
    PassThrough, PassThrough, TypeCast, GemmSpec,
    256, 32, 64, 64, 8, 8, 16, 16, 2, 1,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
    2, 1, S<1, 32, 1, 8>, S<8, 1, 8>,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1,
    ck::Activation::silu_and_mul, false, true, false, true, ck::index_t, BF16>;

using DeviceOpStage2 = ck::tensor_operation::device::DeviceMoeGemm<
    Row, Col, DsLayout, Row, BF16, BF16, DsDataType, BF16, F32, F32,
    PassThrough, PassThrough, TypeCast, GemmSpec,
    256, 64, 128, 64, 8, 8, 16, 16, 4, 2,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
    S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 8, 8, 0,
    2, 2, S<1, 8, 1, 32>, S<2, 1, 2, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3,
    0, false, false, true, true, ck::index_t, BF16>;
// clang-format on

using DeviceOpInstance = std::conditional_t<IsStage1, DeviceOpStage1, DeviceOpStage2>;

static int round_up(int x, int m) { return (x + m - 1) / m * m; }
static float to_f32(BF16 v) { return ck::type_convert<float>(v); }

// [N][K] row-major -> B-preshuffle layout [N0][K0][KLane][NLane][KPack].
static void preshuffle(const BF16* src, BF16* dst, int N, int K)
{
    const int K0 = K / (KLane * KPack);
    for(int n = 0; n < N; ++n)
    {
        for(int k = 0; k < K; ++k)
        {
            const int n0 = n / NPerXdl, n1 = n % NPerXdl;
            const int k0 = k / (KLane * KPack), kt = k % (KLane * KPack);
            const int k1 = kt / KPack, k2 = kt % KPack;
            dst[n0 * KPack * NPerXdl * KLane * K0 + k0 * KPack * NPerXdl * KLane +
                k1 * KPack * NPerXdl + n1 * KPack + k2] = src[n * K + k];
        }
    }
}

int main(int argc, char* argv[])
{
    bool do_verification = true;
    bool time_kernel     = false;
    int tokens           = 333;
    int model_dim        = 200; // not a multiple of the tile or of the preshuffle tile
    int inter            = 256;
    const int topk       = 2;
    const int experts    = 8;

    if(argc == 6)
    {
        do_verification = std::stoi(argv[1]);
        time_kernel     = std::stoi(argv[2]);
        tokens          = std::stoi(argv[3]);
        model_dim       = std::stoi(argv[4]);
        inter           = std::stoi(argv[5]);
    }
    else if(argc != 1)
    {
        std::cout << "arg1: verification (0=no, 1=yes)\n"
                  << "arg2: time kernel (0=no, 1=yes)\n"
                  << "arg3 to 5: tokens, model_dim, inter\n";
        return 1;
    }

    // Stage 1: N = inter (per gate/up half), K = model_dim. Stage 2: N = model_dim, K = inter.
    const int N      = IsStage1 ? inter : model_dim;
    const int K      = IsStage1 ? model_dim : inter;
    const int NPad   = round_up(N, NPerXdl);
    const int KPad   = round_up(K, KLane * KPack);
    const int halves = IsStage1 ? 2 : 1;

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);

    // Routing: distinct experts per token, positive weights normalized per token.
    std::vector<int> topk_ids(tokens * topk);
    std::vector<float> topk_weights(tokens * topk);
    for(int t = 0; t < tokens; ++t)
    {
        std::vector<int> perm(experts);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        float sum = 0.f;
        for(int j = 0; j < topk; ++j)
        {
            topk_ids[t * topk + j]     = perm[j];
            topk_weights[t * topk + j] = 0.1f + std::fabs(uni(rng));
            sum += topk_weights[t * topk + j];
        }
        for(int j = 0; j < topk; ++j)
            topk_weights[t * topk + j] /= sum;
    }

    // Sorted ids: per expert, (topk slot << 24 | token), padded to MPerBlock with token = tokens.
    std::vector<ck::index_t> sorted_token_ids, sorted_expert_ids;
    std::vector<float> sorted_weights;
    for(int e = 0; e < experts; ++e)
    {
        std::vector<std::pair<int, int>> routed;
        for(int t = 0; t < tokens; ++t)
            for(int j = 0; j < topk; ++j)
                if(topk_ids[t * topk + j] == e)
                    routed.emplace_back(t, j);
        if(routed.empty())
            continue;
        const int padded = round_up(static_cast<int>(routed.size()), MPerBlock);
        for(int i = 0; i < padded; ++i)
        {
            const bool valid = i < static_cast<int>(routed.size());
            sorted_token_ids.push_back(valid ? (routed[i].second << 24) | routed[i].first : tokens);
            sorted_weights.push_back(valid ? topk_weights[routed[i].first * topk + routed[i].second]
                                           : 0.f);
        }
        sorted_expert_ids.insert(sorted_expert_ids.end(), padded / MPerBlock, e);
    }
    const int sorted_size = static_cast<int>(sorted_token_ids.size());
    std::vector<ck::index_t> max_token_id(1 + experts, 0);
    max_token_id[0] = sorted_size;

    const int a_rows   = IsStage1 ? tokens : tokens * topk;
    const int out_rows = IsStage1 ? tokens * topk : tokens;
    std::vector<BF16> a(static_cast<size_t>(a_rows) * K);
    for(auto& v : a)
        v = ck::type_convert<BF16>(uni(rng));
    std::vector<BF16> b(static_cast<size_t>(experts) * halves * N * K);
    for(auto& v : b)
        v = ck::type_convert<BF16>(0.25f * uni(rng));

    // Zero-pad each expert (and gate/up half) to NPad x KPad, then preshuffle.
    std::vector<BF16> b_padded(static_cast<size_t>(experts) * halves * NPad * KPad,
                               ck::type_convert<BF16>(0.f));
    for(int eh = 0; eh < experts * halves; ++eh)
        for(int n = 0; n < N; ++n)
            std::memcpy(&b_padded[(static_cast<size_t>(eh) * NPad + n) * KPad],
                        &b[(static_cast<size_t>(eh) * N + n) * K],
                        K * sizeof(BF16));
    std::vector<BF16> b_preshuffled(b_padded.size());
    for(int e = 0; e < experts; ++e)
        preshuffle(&b_padded[static_cast<size_t>(e) * halves * NPad * KPad],
                   &b_preshuffled[static_cast<size_t>(e) * halves * NPad * KPad],
                   halves * NPad,
                   KPad);

    DeviceMem sorted_token_ids_dev(sizeof(ck::index_t) * sorted_token_ids.size());
    DeviceMem sorted_expert_ids_dev(sizeof(ck::index_t) * sorted_expert_ids.size());
    DeviceMem max_token_id_dev(sizeof(ck::index_t) * max_token_id.size());
    DeviceMem sorted_weights_dev(sizeof(float) * sorted_weights.size());
    DeviceMem a_dev(sizeof(BF16) * a.size());
    DeviceMem b_dev(sizeof(BF16) * b_preshuffled.size());
    DeviceMem e_dev(sizeof(BF16) * out_rows * N);
    sorted_token_ids_dev.ToDevice(sorted_token_ids.data());
    sorted_expert_ids_dev.ToDevice(sorted_expert_ids.data());
    max_token_id_dev.ToDevice(max_token_id.data());
    sorted_weights_dev.ToDevice(sorted_weights.data());
    a_dev.ToDevice(a.data());
    b_dev.ToDevice(b_preshuffled.data());
    e_dev.SetZero(); // stage 2 accumulates with atomic adds

    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(
        sorted_token_ids_dev.GetDeviceBuffer(),
        sorted_expert_ids_dev.GetDeviceBuffer(),
        max_token_id_dev.GetDeviceBuffer(),
        a_dev.GetDeviceBuffer(),
        b_dev.GetDeviceBuffer(),
        std::array<const void*, 3>{
            nullptr, nullptr, IsStage1 ? nullptr : sorted_weights_dev.GetDeviceBuffer()},
        e_dev.GetDeviceBuffer(),
        tokens,
        topk,
        sorted_size,
        N,
        K,
        K,
        K,
        std::array<ck::index_t, 3>{1, 1, 0},
        N,
        1,
        PassThrough{},
        PassThrough{},
        TypeCast{});

    if(!device_op.IsSupportedArgument(argument))
    {
        std::cout << device_op.GetTypeString() << " does not support this problem" << std::endl;
        return 1;
    }

    invoker.Run(argument, StreamConfig{nullptr, false});

    int rc = 0;
    if(do_verification)
    {
        std::vector<BF16> e_device(static_cast<size_t>(out_rows) * N);
        e_dev.FromDevice(e_device.data());

        std::vector<float> e_ref(e_device.size(), 0.f);
        for(int t = 0; t < tokens; ++t)
        {
            for(int j = 0; j < topk; ++j)
            {
                const int e = topk_ids[t * topk + j];
                for(int n = 0; n < N; ++n)
                {
                    if constexpr(IsStage1)
                    {
                        const BF16* x  = &a[static_cast<size_t>(t) * K];
                        const BF16* wg = &b[((static_cast<size_t>(e) * 2) * N + n) * K];
                        const BF16* wu = &b[((static_cast<size_t>(e) * 2 + 1) * N + n) * K];
                        float gate = 0.f, up = 0.f;
                        for(int k = 0; k < K; ++k)
                        {
                            gate += to_f32(x[k]) * to_f32(wg[k]);
                            up += to_f32(x[k]) * to_f32(wu[k]);
                        }
                        e_ref[(static_cast<size_t>(t) * topk + j) * N + n] =
                            gate / (1.f + std::exp(-gate)) * up;
                    }
                    else
                    {
                        const BF16* x  = &a[(static_cast<size_t>(t) * topk + j) * K];
                        const BF16* wd = &b[(static_cast<size_t>(e) * N + n) * K];
                        float acc      = 0.f;
                        for(int k = 0; k < K; ++k)
                            acc += to_f32(x[k]) * to_f32(wd[k]);
                        e_ref[static_cast<size_t>(t) * N + n] += topk_weights[t * topk + j] * acc;
                    }
                }
            }
        }
        std::vector<BF16> e_host(e_ref.size());
        std::transform(e_ref.begin(), e_ref.end(), e_host.begin(), [](float v) {
            return ck::type_convert<BF16>(v);
        });
        rc =
            ck::utils::check_err(e_device, e_host, "Error: Incorrect results!", 2e-2, 5e-2) ? 0 : 1;
    }

    if(time_kernel)
    {
        const float ave_time = invoker.Run(argument, StreamConfig{nullptr, true});
        std::cout << "Perf: " << ave_time << " ms, " << device_op.GetTypeString() << std::endl;
    }
    return rc;
}
