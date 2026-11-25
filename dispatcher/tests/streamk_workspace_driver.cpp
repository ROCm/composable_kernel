// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Compiled by test_streamk_gpu_correctness.py with one generated SelectedKernel.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(hipError_t e)
{
    if(e != hipSuccess)
        throw std::runtime_error(hipGetErrorString(e));
}

struct Buffer
{
    static constexpr size_t guard = 256;
    unsigned char* base           = nullptr;
    size_t bytes;
    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;
    explicit Buffer(size_t n) : bytes(n) { check(hipMalloc(&base, bytes + 2 * guard)); }
    ~Buffer() { (void)hipFree(base); }
    void* data() { return base + guard; }
    void poison() { check(hipMemset(base, 0xa5, bytes + 2 * guard)); }
    void verify_guards()
    {
        std::vector<unsigned char> h(2 * guard);
        check(hipMemcpy(h.data(), base, guard, hipMemcpyDeviceToHost));
        check(hipMemcpy(h.data() + guard, base + guard + bytes, guard, hipMemcpyDeviceToHost));
        if(!std::all_of(h.begin(), h.end(), [](auto v) { return v == 0xa5; }))
            throw std::runtime_error("buffer guard changed");
    }
};

void run_case(const std::string& name, int m, int n, int k, int budget, bool want_sk)
{
    const int ldc = n + 16;
    Buffer a(size_t(m) * k * sizeof(ADataType));
    Buffer b(size_t(n) * k * sizeof(BDataType));
    Buffer c(size_t(m) * ldc * sizeof(CDataType));
    ck_tile::StreamKHostArgs args(a.data(), b.data(), c.data(), m, n, k, k, k, ldc);
    auto ka        = SelectedKernel::MakeKargs(args, budget);
    const auto& tp = ka.tile_partitioner;
    auto ws_bytes  = SelectedKernel::GetWorkSpaceSize(args, budget);
    std::cout << "CASE " << name << " m=" << m << " n=" << n << " k=" << k << " budget=" << budget
              << " wgs=" << tp.get_max_active_wgs() << " sk_ctas=" << tp.get_sk_ctas()
              << " flags=" << tp.get_flags_buffer_size() << " workspace=" << ws_bytes << std::endl;
    if(!SelectedKernel::IsSupported(args, budget))
        throw std::runtime_error("intended supported shape rejected");
    if((tp.get_sk_ctas() > 0) != want_sk)
        throw std::runtime_error("did not reach intended partitioner branch");
    if(!want_sk && (ws_bytes != 0 || tp.get_flags_buffer_size() != 0))
        throw std::runtime_error("Full-DP unexpectedly needs workspace");
    Buffer workspace(ws_bytes);
    if(ws_bytes)
    {
        bool rejected = false;
        try
        {
            SelectedKernel::launch(args, ck_tile::stream_config{}, nullptr, budget);
        }
        catch(const std::invalid_argument&)
        {
            rejected = true;
        }
        if(!rejected)
            throw std::runtime_error("null external workspace accepted");
    }
    std::vector<ADataType> ah(size_t(m) * k);
    std::vector<BDataType> bh(size_t(n) * k);
    std::vector<CDataType> ch(size_t(m) * ldc);
    for(int phase = 0; phase < 3; ++phase)
    {
        // Products are multiples of 1/64. Every 35-element period sums to zero,
        // so partial sums and outputs stay exactly representable in bf16. This
        // gives a strict check even for long K; phase changes catch stale C.
        float refs[7][5] = {};
        for(int r = 0; r < 7; ++r)
            for(int s = 0; s < 5; ++s)
                for(int x = 0; x < k; ++x)
                    refs[r][s] += float((r + x + phase) % 7 - 3) *
                                  float((3 * x + s + 2 * phase) % 5 - 2) / 64.0f;
        for(int r = 0; r < m; ++r)
            for(int x = 0; x < k; ++x)
                ah[size_t(r) * k + x] =
                    ck_tile::type_convert<ADataType>(float((r + x + phase) % 7 - 3) / 8.0f);
        for(int s = 0; s < n; ++s)
            for(int x = 0; x < k; ++x)
                bh[size_t(s) * k + x] =
                    ck_tile::type_convert<BDataType>(float((3 * x + s + 2 * phase) % 5 - 2) / 8.0f);
        check(hipMemcpy(a.data(), ah.data(), a.bytes, hipMemcpyHostToDevice));
        check(hipMemcpy(b.data(), bh.data(), b.bytes, hipMemcpyHostToDevice));
        // Reuse the allocation with stale flags/partials, guarded on both ends.
        c.poison();
        workspace.poison();
        SelectedKernel::launch(
            args, ck_tile::stream_config{}, ws_bytes ? workspace.data() : nullptr, budget);
        check(hipGetLastError());
        check(hipDeviceSynchronize());
        check(hipMemcpy(ch.data(), c.data(), c.bytes, hipMemcpyDeviceToHost));
        size_t bad      = 0;
        float max_error = 0;
        for(int r = 0; r < m; ++r)
            for(int s = 0; s < n; ++s)
            {
                float got = ck_tile::type_convert<float>(ch[size_t(r) * ldc + s]);
                float err = std::abs(got - refs[r % 7][s % 5]);
                max_error = std::max(max_error, err);
                bad += !std::isfinite(got) || err != 0.0f;
            }
        auto* raw = reinterpret_cast<unsigned char*>(ch.data());
        for(int r = 0; r < m; ++r)
            for(size_t x = size_t(r * ldc + n) * sizeof(CDataType);
                x < size_t((r + 1) * ldc) * sizeof(CDataType);
                ++x)
                if(raw[x] != 0xa5)
                    throw std::runtime_error("C row padding changed");
        c.verify_guards();
        workspace.verify_guards();
        std::cout << "RESULT phase=" << phase << " bad=" << bad << " max_error=" << max_error
                  << std::endl;
        if(bad)
            throw std::runtime_error("numerical mismatch");
    }
}

int main()
{
    try
    {
        constexpr int tm = SelectedKernel::TileM;
        constexpr int tn = SelectedKernel::TileN;
        constexpr int tk = SelectedKernel::TileK;
        ck_tile::StreamKHostArgs probe(nullptr, nullptr, nullptr, tm, tn, tk, tk, tk, tn);
        int full = SelectedKernel::MakeKargs(probe).tile_partitioner.get_max_active_wgs();
        for(int pct : {0, -1, 101})
        {
            bool rejected = false;
            try
            {
                (void)SelectedKernel::MakeKargs(probe, pct);
            }
            catch(const std::invalid_argument&)
            {
                rejected = true;
            }
            if(!rejected)
                throw std::runtime_error("invalid budget accepted");
        }
        // A single tile with enough K work must use Stream-K. An exact multiple
        // of the budget capacity must use Full-DP with zero workspace/flags.
        for(int pct : {100, 50, 33})
        {
            int wgs = std::max(1, full * pct / 100);
            if(SelectedKernel::MakeKargs(probe, pct).tile_partitioner.get_max_active_wgs() != wgs)
                throw std::runtime_error("wrong budget capacity");
            run_case("active_sk", tm, tn, tk * (full + 5) + 13, pct, wgs > 1);
            run_case("full_dp", tm, tn * wgs, tk + 13, pct, false);
        }
        run_case("budget_one", tm, tn * std::max(1, full / 100), tk + 13, 1, false);
        run_case("low_work", tm, tn, 1, 100, false);
        std::cout << "PASS all 8 cases, 24 launches, guards and negative API checks" << std::endl;
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "FAIL " << e.what() << std::endl;
        return 1;
    }
}
