// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <vector>
#include <cassert>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/utility/flush_icache.hpp"
namespace ck {
namespace utility {

template <typename Argument, typename DsDataType>
struct RotatingMemWrapperMultiD
{
    static constexpr index_t NumDs = DsDataType::Size();

    using ADataType     = decltype(Argument::p_a_grid);
    using BDataType     = decltype(Argument::p_b_grid);
    using DsGridPointer = decltype(Argument::p_ds_grid);

    RotatingMemWrapperMultiD() = delete;
    RotatingMemWrapperMultiD(Argument& arg_,
                             std::size_t rotating_count_,
                             std::size_t size_a_,
                             std::size_t size_b_,
                             std::array<std::size_t, NumDs> size_ds_)
        : arg(arg_),
          rotating_count(rotating_count_),
          size_a(size_a_),
          size_b(size_b_),
          size_ds(size_ds_)
    {
        p_a_grids.push_back(arg.p_a_grid);
        p_b_grids.push_back(arg.p_b_grid);
        p_ds_grids.push_back(arg.p_ds_grid);
        for(size_t i = 1; i < rotating_count; i++)
        {
            {
                void* pADeviceBuf;
                hip_check_error(hipMalloc(static_cast<void**>(&pADeviceBuf), size_a_));
                hip_check_error(hipMemcpy(static_cast<void*>(pADeviceBuf),
                                          const_cast<void*>(p_a_grids[0]),
                                          size_a_,
                                          hipMemcpyDeviceToDevice));
                p_a_grids.push_back(pADeviceBuf);
            }

            {
                void* pBDeviceBuf;
                hip_check_error(hipMalloc(static_cast<void**>(&pBDeviceBuf), size_b_));
                hip_check_error(hipMemcpy(static_cast<void*>(pBDeviceBuf),
                                          const_cast<void*>(p_b_grids[0]),
                                          size_b_,
                                          hipMemcpyDeviceToDevice));
                p_b_grids.push_back(pBDeviceBuf);
            }

            {

                DsGridPointer ds_buffer;
                static_for<0, NumDs, 1>{}([&](auto j) {
                    void* pDDeviceBuf;
                    hip_check_error(hipMalloc(static_cast<void**>(&pDDeviceBuf), size_ds_[j]));
                    hip_check_error(hipMemcpy(static_cast<void*>(pDDeviceBuf),
                                              static_cast<const void*>(p_ds_grids[0][j]),
                                              size_ds_[j],
                                              hipMemcpyDeviceToDevice));

                    using DDataType = remove_cvref_t<tuple_element_t<j.value, DsDataType>>;

                    ds_buffer(j) = static_cast<const DDataType*>(pDDeviceBuf);
                });

                p_ds_grids.push_back(ds_buffer);
            }
        }
    }

    void Next()
    {
        if(rotating_count > 1)
        {
            std::size_t idx = iter++ % rotating_count;
            arg.p_a_grid    = reinterpret_cast<ADataType>(p_a_grids[idx]);
            arg.p_b_grid    = reinterpret_cast<BDataType>(p_b_grids[idx]);
            arg.p_ds_grid   = p_ds_grids[idx];
        }
    }
    void Print()
    {
        std::cout << "RotatingMemWrapperMultiD: { size_a: " << size_a << ", size_b: " << size_b
                  << ", rotating_count: " << rotating_count << "}" << std::endl;
    }
    ~RotatingMemWrapperMultiD()
    {
        if(rotating_count > 1)
        {
            // restore ptr
            arg.p_a_grid  = reinterpret_cast<ADataType>(p_a_grids[0]);
            arg.p_b_grid  = reinterpret_cast<BDataType>(p_b_grids[0]);
            arg.p_ds_grid = p_ds_grids[0];

            // free device mem
            for(size_t i = 1; i < rotating_count; i++)
            {
                hip_check_error(hipFree(const_cast<void*>(p_a_grids[i])));
                hip_check_error(hipFree(const_cast<void*>(p_b_grids[i])));

                static_for<0, NumDs, 1>{}([&](auto j) {
                    using DDataType = remove_cvref_t<tuple_element_t<j.value, DsDataType>>;
                    hip_check_error(
                        hipFree(static_cast<void*>(const_cast<DDataType*>(p_ds_grids[i][j]))));
                });
            }
        }
    }

    private:
    Argument& arg;
    std::size_t iter                       = 0;
    std::size_t rotating_count             = 1;
    std::size_t size_a                     = 0;
    std::size_t size_b                     = 0;
    std::array<std::size_t, NumDs> size_ds = {0};
    std::vector<const void*> p_a_grids;
    std::vector<const void*> p_b_grids;
    std::vector<DsGridPointer> p_ds_grids;
};

template <typename Argument>
struct RotatingMemWrapper
{
    using ADataType = decltype(Argument::p_a_grid);
    using BDataType = decltype(Argument::p_b_grid);

    RotatingMemWrapper() = delete;
    RotatingMemWrapper(Argument& arg_,
                       std::size_t rotating_count_,
                       std::size_t size_a_,
                       std::size_t size_b_)
        : arg(arg_), rotating_count(rotating_count_), size_a(size_a_), size_b(size_b_)
    {
        p_a_grids.push_back(arg.p_a_grid);
        p_b_grids.push_back(arg.p_b_grid);
        for(size_t i = 1; i < rotating_count; i++)
        {
            {
                void* pADeviceBuf;
                hip_check_error(hipMalloc(static_cast<void**>(&pADeviceBuf), size_a_));
                hip_check_error(hipMemcpy(static_cast<void*>(pADeviceBuf),
                                          const_cast<void*>(p_a_grids[0]),
                                          size_a_,
                                          hipMemcpyDeviceToDevice));
                p_a_grids.push_back(pADeviceBuf);
            }

            {
                void* pBDeviceBuf;
                hip_check_error(hipMalloc(static_cast<void**>(&pBDeviceBuf), size_b_));
                hip_check_error(hipMemcpy(static_cast<void*>(pBDeviceBuf),
                                          const_cast<void*>(p_b_grids[0]),
                                          size_b_,
                                          hipMemcpyDeviceToDevice));
                p_b_grids.push_back(pBDeviceBuf);
            }
        }
    }

    void Next()
    {
        if(rotating_count > 1)
        {
            std::size_t idx = iter++ % rotating_count;
            arg.p_a_grid    = reinterpret_cast<ADataType>(p_a_grids[idx]);
            arg.p_b_grid    = reinterpret_cast<BDataType>(p_b_grids[idx]);
        }
    }
    void Print()
    {
        std::cout << "RotatingMemWrapper: { size_a: " << size_a << ", size_b: " << size_b
                  << ", rotating_count: " << rotating_count << "}" << std::endl;
    }
    ~RotatingMemWrapper()
    {
        if(rotating_count > 1)
        {
            // restore ptr
            arg.p_a_grid = reinterpret_cast<ADataType>(p_a_grids[0]);
            arg.p_b_grid = reinterpret_cast<BDataType>(p_b_grids[0]);

            // free device mem
            for(size_t i = 1; i < rotating_count; i++)
            {
                hip_check_error(hipFree(const_cast<void*>(p_a_grids[i])));
                hip_check_error(hipFree(const_cast<void*>(p_b_grids[i])));
            }
        }
    }

    private:
    Argument& arg;
    std::size_t iter           = 0;
    std::size_t rotating_count = 1;
    std::size_t size_a         = 0;
    std::size_t size_b         = 0;
    std::vector<const void*> p_a_grids;
    std::vector<const void*> p_b_grids;
};

inline void flush_icache()
{
    hipDeviceProp_t deviceProps;
    hip_check_error(hipGetDeviceProperties(&deviceProps, 0));
    int32_t gpu_block3 = deviceProps.multiProcessorCount * 60;

    ck::flush_icache<<<dim3(gpu_block3), dim3(64), 0, nullptr>>>();
    hip_check_error(hipGetLastError());
}
// if TimePrePress == false, return time does not include preprocess's time
// XXX This is not used?
template <bool TimePreprocess,
          typename GemmArgs,
          typename... Args,
          typename F,
          typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess(const StreamConfig& stream_config,
                                             PreProcessFunc preprocess,
                                             F kernel,
                                             dim3 grid_dim,
                                             dim3 block_dim,
                                             std::size_t lds_byte,
                                             GemmArgs& gemm_args,
                                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: kernel %p, grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
                   __func__,
                   reinterpret_cast<void*>(kernel),
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }
        if(!ck::EnvIsEnabled(CK_ENV(CK_NEWTIMING)))
        {
            // old algorithm: warm up then time nrepeat iterations and take the mean
            // warm up
            for(int i = 0; i < stream_config.cold_niters_; ++i)
            {
                kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(gemm_args,
                                                                                    args...);
                hip_check_error(hipGetLastError());
            }
        }

        const int nrepeat = stream_config.nrepeat_;
        assert(nrepeat > 0);

        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Start running %d times...\n", nrepeat);
        }

        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        if(!ck::EnvIsEnabled(CK_ENV(CK_NEWTIMING)))
        {
            // Time the entire loop
            hip_check_error(hipEventRecord(start, stream_config.stream_id_));
            for(int i = 0; i < nrepeat; ++i)
            {
                preprocess();

                kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(gemm_args,
                                                                                    args...);
                hip_check_error(hipGetLastError());
            }
            hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(stop));

            float total_time = 0;
            hip_check_error(hipEventElapsedTime(&total_time, start, stop));

            // return total_time / nrepeat adjusted by supposed cost of icache clear
            hipDeviceProp_t deviceProps;
            hip_check_error(hipGetDeviceProperties(&deviceProps, 0));
            // This seems pretty random
            float preprocess_offset = deviceProps.multiProcessorCount == 80 ? 0.005 : 0.01;
            return (total_time - preprocess_offset * nrepeat) / nrepeat;
        }
        // New algorithm: adaptive, no warmup, return minimum
        // Need to print all the values for stats.
        constexpr float maxtime = 10;   // ms
        constexpr int minrep = 3;
        std::vector<float> times;
        float total_time = 0;
        hip_check_error(hipDeviceSynchronize());
        for (int i = 0; i < nrepeat; ++i)
        {
            hipExtLaunchKernelGGL(kernel,
                                  grid_dim,
                                  block_dim,
                                  lds_byte,
                                  stream_config.stream_id_,
                                  start,
                                  stop,
                                  0,
                                  gemm_args,
                                  args...);
            hip_check_error(hipGetLastError());
            hip_check_error(hipEventSynchronize(stop));

            float cur_time = 0;
            hip_check_error(hipEventElapsedTime(&cur_time, start, stop));
            times.push_back(cur_time);
            total_time += cur_time;
            if(total_time >= maxtime && i + 1 >= minrep)
                break;
        }
        printf("%s: kernel %p, grid_dim {%u, %u, %u}, block_dim {%u, %u, %u}, ",
               __func__,
               reinterpret_cast<void *>(kernel),
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z);
        printf("times {");
        for (float t: times) printf(" %.7f", t);
        printf("}\n");
        return total_time / times.size(); //*std::min_element(times.begin(), times.end());
    }
    else
    {
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(gemm_args, args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(gemm_args, args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}

} // namespace utility
} // namespace ck
