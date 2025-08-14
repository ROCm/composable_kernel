// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#ifndef __HIPCC_RTC__
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"


template <class it>
typename std::iterator_traits<it>::value_type median(it begin, it end)
{
    if(begin == end)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    auto n  = std::distance(begin, end);
    auto n2 = n / 2;
    std::nth_element(begin, begin + n2, end);
    return (n % 2) ? begin[n2] : (*std::max_element(begin, begin + n2) + begin[n2]) / 2.0;
}

inline void remove_outliers(std::vector<float>& v)
{
    // 1.5x IQR method to detect and remove outliers
    auto n2 = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n2, v.end());
    auto q1  = median(v.begin(), v.begin() + n2);
    auto q3  = median(v.begin() + ((v.size() % 2) ? n2 + 1 : n2), v.end());
    auto iqr = q3 - q1;
    auto lb  = q1 - 1.5 * iqr;
    auto ub  = q3 + 1.5 * iqr;
    v.erase(std::remove_if(v.begin(), v.end(), [&](float f) { return f < lb || f > ub; }), v.end());
}


template <typename... Args, typename F>
float launch_and_time_kernel(const StreamConfig& stream_config,
                             F kernel,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             Args... args)
{

    printf("*******************************************************************\n");
    printf("Launching kernel: %s\n", __func__);
    printf("*******************************************************************\n");

#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
                   __func__,
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }
        // warm up
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        const int nrepeat = stream_config.nrepeat_;
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Start running %d times...\n", nrepeat);
        }
        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(start, stream_config.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));

        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));

        hip_check_error(hipEventDestroy(start));
        hip_check_error(hipEventDestroy(stop));

        return total_time / nrepeat;
    }
    else
    {
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}

template <typename... Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess(const StreamConfig& stream_config,
                                             PreProcessFunc preprocess,
                                             F kernel,
                                             dim3 grid_dim,
                                             dim3 block_dim,
                                             std::size_t lds_byte,
                                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
                   __func__,
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }
        // warm up
        preprocess();
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        const int nrepeat = stream_config.nrepeat_;
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Start running %d times...\n", nrepeat);
        }
        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(start, stream_config.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            preprocess();
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));

        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));

        hip_check_error(hipEventDestroy(start));
        hip_check_error(hipEventDestroy(stop));

        return total_time / nrepeat;
    }
    else
    {
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}

// template <typename... Args, typename F>
// float launch_and_time_kernel_tb(const StreamConfig& stream_config,
//                              F kernel,
//                              dim3 grid_dim,
//                              dim3 block_dim,
//                              std::size_t lds_byte,
//                              Args... args)
// {

//     printf("*******************************************************************\n");
//     printf(" Kernel is running : %s\n", __func__);
//     printf("*******************************************************************\n");

// #if CK_TIME_KERNEL
//     if(stream_config.time_kernel_)
//     {
//         if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
//         {
//             printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
//                    __func__,
//                    grid_dim.x,
//                    grid_dim.y,
//                    grid_dim.z,
//                    block_dim.x,
//                    block_dim.y,
//                    block_dim.z);

//             printf("Warm up %d times\n", stream_config.cold_niters_);
//         }
//         // warm up
//         for(int i = 0; i < stream_config.cold_niters_; ++i)
//         {
//             kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
//             hip_check_error(hipGetLastError());
//         }

//         const int nrepeat = stream_config.nrepeat_;
//         if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
//         {
//             printf("Start running %d times...\n", nrepeat);
//         }
        
//         std::vector<hipEvent_t> start_event(2), stop_event(2);
//         for(auto& e : start_event)
//         {
//             hip_check_error(hipEventCreate(&e));
//         }
//         for(auto& e : stop_event)
//         {
//             hip_check_error(hipEventCreate(&e));
//         }
//         hipEvent_t event0;
//         hip_check_error(hipEventCreate(&event0));
//         hip_check_error(hipEventRecord(event0, stream_config.stream_id_));

//         float per_iter_time = 0.f;
//         std::vector<float> times;
//         int i                     = 0;
//         float gpu_time_used       = 0.;
//         float bench_time_secs     = 0.f; // need to pass as command line argument
//         const float bench_time_ms = bench_time_secs * 1000;

//         while(i < nrepeat || per_iter_time < bench_time_ms)
//         {
          
//             hip_check_error(hipEventRecord(start_event[i % 2], stream_config.stream_id_));
//             kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(gemm_args, args...);
//             hip_check_error(hipEventRecord(stop_event[i % 2], stream_config.stream_id_));

//             if(i > 0)
//             {
//                 // while iteration i is ongoing, wait for iteration i-1 to end
//                 hip_check_error(hipEventSynchronize(stop_event[(i - 1) % 2]));
//                 hip_check_error(hipEventElapsedTime(
//                     &per_iter_time, start_event[(i - 1) % 2], stop_event[(i - 1) % 2]));
//                 // record time for iteration i-1
//                 times.push_back(per_iter_time);
//                 // if iterations 0 to i-1 took more than the required runtime, we can stop
//                 hip_check_error(
//                     hipEventElapsedTime(&per_iter_time, event0, stop_event[(i - 1) % 2]));
//             }
//             i++;
//         }
//         if(!i)
//         {
//             gpu_time_used = 0.;
//         }
//         else
//         {
//             // wait for the final iteration
//             hip_check_error(hipEventSynchronize(stop_event[(i - 1) % 2]));
//             hip_check_error(hipEventElapsedTime(
//                 &per_iter_time, start_event[(i - 1) % 2], stop_event[(i - 1) % 2]));
//             times.push_back(per_iter_time);
//             remove_outliers(times);
//             gpu_time_used = std::accumulate(times.begin(), times.end(), 0.) / times.size();
//             printf("Accumulate time per iteration: %f ms\n",
//                    std::accumulate(times.begin(), times.end(), 0.));
//             printf("total time after removing outliers: %zu \n", times.size());
//             // gpu_time_used *= 1000; // ms to us
//         }
//         return gpu_time_used;

       
//     }
//     else
//     {
//         kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
//         hip_check_error(hipGetLastError());

//         return 0;
//     }
// #else
//     kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
//     hip_check_error(hipGetLastError());

//     return 0;
// #endif
// }


template <typename... Args, typename F>
float launch_and_time_kernel_tb(const StreamConfig& stream_config,
                             F kernel,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             float cold_bench_time_secs,
                             float hot_bench_time_secs,
                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u}\n",
                   __func__,
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Cold run time: %.3f seconds\n", cold_bench_time_secs);
            printf("Hot run time: %.3f seconds\n", hot_bench_time_secs);
        }

        // Create events for timing
        hipEvent_t cold_start, cold_end, hot_start, hot_end, iter_start, iter_end;
        hip_check_error(hipEventCreate(&cold_start));
        hip_check_error(hipEventCreate(&cold_end));
        hip_check_error(hipEventCreate(&hot_start));
        hip_check_error(hipEventCreate(&hot_end));
        hip_check_error(hipEventCreate(&iter_start));
        hip_check_error(hipEventCreate(&iter_end));

        // Convert seconds to milliseconds for HIP API
        const float cold_time_ms = cold_bench_time_secs * 1000.0f;
        const float hot_time_ms = hot_bench_time_secs * 1000.0f;
        
        // ======== COLD RUN (WARM-UP) ========
        float elapsed_cold_ms = 0.0f;
        int cold_iterations = 0;
        
        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(cold_start, stream_config.stream_id_));
        
        while(elapsed_cold_ms < cold_time_ms)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
            cold_iterations++;
            
            // Check elapsed time periodically to reduce overhead
            if(cold_iterations % 5 == 0)
            {
                hip_check_error(hipEventRecord(cold_end, stream_config.stream_id_));
                hip_check_error(hipEventSynchronize(cold_end));
                hip_check_error(hipEventElapsedTime(&elapsed_cold_ms, cold_start, cold_end));
            }
        }
        
        // Get final cold run time
        hip_check_error(hipEventRecord(cold_end, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(cold_end));
        hip_check_error(hipEventElapsedTime(&elapsed_cold_ms, cold_start, cold_end));
        
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Cold run completed: %d iterations in %.3f ms (%.3f ms/iter)\n", 
                   cold_iterations, elapsed_cold_ms, elapsed_cold_ms / cold_iterations);
        }
        
        // ======== HOT RUN (MEASUREMENT) ========
        std::vector<float> iter_times;
        float elapsed_hot_ms = 0.0f;
        int hot_iterations = 0;
        
        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(hot_start, stream_config.stream_id_));
        
        while(elapsed_hot_ms < hot_time_ms)
        {
            // Time individual iterations for statistics
            hip_check_error(hipEventRecord(iter_start, stream_config.stream_id_));
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipEventRecord(iter_end, stream_config.stream_id_));
            
            // Wait for iteration to complete and record its time
            hip_check_error(hipEventSynchronize(iter_end));
            float iter_time = 0.0f;
            hip_check_error(hipEventElapsedTime(&iter_time, iter_start, iter_end));
            iter_times.push_back(iter_time);
            
            hot_iterations++;
            
            // Update total elapsed time
            hip_check_error(hipEventRecord(hot_end, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(hot_end));
            hip_check_error(hipEventElapsedTime(&elapsed_hot_ms, hot_start, hot_end));
        }
        
        // Process timing statistics
        if(iter_times.size() > 4) // Only remove outliers if we have enough samples
        {
            remove_outliers(iter_times);
        }
        
        float avg_time = 0.0f;
        if(!iter_times.empty())
        {
            avg_time = std::accumulate(iter_times.begin(), iter_times.end(), 0.0f) / 
                       iter_times.size();
        }
        
        // Print performance statistics
        printf("Performance statistics:\n");
        printf("  Total hot run time: %.3f ms\n", elapsed_hot_ms);
        printf("  Hot iterations: %d (used %zu after removing outliers)\n", 
               hot_iterations, iter_times.size());
        printf("  Average time per iteration: %.3f ms\n", avg_time);
        
        // Clean up events
        hip_check_error(hipEventDestroy(cold_start));
        hip_check_error(hipEventDestroy(cold_end));
        hip_check_error(hipEventDestroy(hot_start));
        hip_check_error(hipEventDestroy(hot_end));
        hip_check_error(hipEventDestroy(iter_start));
        hip_check_error(hipEventDestroy(iter_end));
        
        return avg_time;
    }
    else
    {
        // Just run the kernel once without timing
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());
        return 0.0f;
    }
#else
    // CK_TIME_KERNEL not defined, just run once
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());
    return 0.0f;
#endif
}

template <typename... Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess_tb(const StreamConfig& stream_config,
                                             PreProcessFunc preprocess,
                                             F kernel,
                                             dim3 grid_dim,
                                             dim3 block_dim,
                                             std::size_t lds_byte,
                                             float cold_bench_time_secs,
                                             float hot_bench_time_secs,
                                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u}\n",
                   __func__,
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Cold run time: %.3f seconds\n", cold_bench_time_secs);
            printf("Hot run time: %.3f seconds\n", hot_bench_time_secs);
        }

        // Create events for timing
        hipEvent_t cold_start, cold_end, hot_start, hot_end, iter_start, iter_end;
        hip_check_error(hipEventCreate(&cold_start));
        hip_check_error(hipEventCreate(&cold_end));
        hip_check_error(hipEventCreate(&hot_start));
        hip_check_error(hipEventCreate(&hot_end));
        hip_check_error(hipEventCreate(&iter_start));
        hip_check_error(hipEventCreate(&iter_end));

        // Convert seconds to milliseconds for HIP API
        const float cold_time_ms = cold_bench_time_secs * 1000.0f;
        const float hot_time_ms = hot_bench_time_secs * 1000.0f;
        
        // ======== COLD RUN (WARM-UP) ========
        float elapsed_cold_ms = 0.0f;
        int cold_iterations = 0;
        
        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(cold_start, stream_config.stream_id_));
        
        while(elapsed_cold_ms < cold_time_ms)
        {
            preprocess();
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
            cold_iterations++;
            
            // Check elapsed time periodically (every 5 iterations to reduce overhead)
            if(cold_iterations % 5 == 0)
            {
                hip_check_error(hipEventRecord(cold_end, stream_config.stream_id_));
                hip_check_error(hipEventSynchronize(cold_end));
                hip_check_error(hipEventElapsedTime(&elapsed_cold_ms, cold_start, cold_end));
            }
        }
        
        // Get final cold run time
        hip_check_error(hipEventRecord(cold_end, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(cold_end));
        hip_check_error(hipEventElapsedTime(&elapsed_cold_ms, cold_start, cold_end));
        
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Cold run: %d iterations in %.3f ms (%.3f ms/iter)\n", 
                   cold_iterations, elapsed_cold_ms, elapsed_cold_ms / cold_iterations);
        }
        
        // ======== HOT RUN (MEASUREMENT) ========
        std::vector<float> iter_times;
        float elapsed_hot_ms = 0.0f;
        int hot_iterations = 0;
        
        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(hot_start, stream_config.stream_id_));
        
        while(elapsed_hot_ms < hot_time_ms)
        {
            // Time individual iterations for statistics
            preprocess();
            
            hip_check_error(hipEventRecord(iter_start, stream_config.stream_id_));
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipEventRecord(iter_end, stream_config.stream_id_));
            
            // Wait for iteration to complete and record its time
            hip_check_error(hipEventSynchronize(iter_end));
            float iter_time = 0.0f;
            hip_check_error(hipEventElapsedTime(&iter_time, iter_start, iter_end));
            iter_times.push_back(iter_time);
            
            hot_iterations++;
            
            // Update total elapsed time
            hip_check_error(hipEventRecord(hot_end, stream_config.stream_id_));
            hip_check_error(hipEventSynchronize(hot_end));
            hip_check_error(hipEventElapsedTime(&elapsed_hot_ms, hot_start, hot_end));
        }
        
        // Process timing statistics
        if(iter_times.size() > 4) // Only remove outliers if we have enough samples
        {
            remove_outliers(iter_times);
        }
        
        float avg_time = 0.0f;
        if(!iter_times.empty())
        {
            avg_time = std::accumulate(iter_times.begin(), iter_times.end(), 0.0f) / 
                       iter_times.size();
        }
        
        // Print performance statistics
        printf("Performance statistics:\n");
        printf("  Total hot run time: %.3f ms\n", elapsed_hot_ms);
        printf("  Iterations: %d (used %zu after removing outliers)\n", 
               hot_iterations, iter_times.size());
        printf("  Average time per iteration: %.3f ms\n", avg_time);
        
        // Clean up events
        hip_check_error(hipEventDestroy(cold_start));
        hip_check_error(hipEventDestroy(cold_end));
        hip_check_error(hipEventDestroy(hot_start));
        hip_check_error(hipEventDestroy(hot_end));
        hip_check_error(hipEventDestroy(iter_start));
        hip_check_error(hipEventDestroy(iter_end));
        
        return avg_time;
    }
    else
    {
        // Just run the kernel once without timing
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());
        return 0.0f;
    }
#else
    // CK_TIME_KERNEL not defined, just run once
    preprocess();
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());
    return 0.0f;
#endif
}

// template <typename... Args, typename F, typename PreProcessFunc>
// float launch_and_time_kernel_with_preprocess_tb(const StreamConfig& stream_config,
//                                              PreProcessFunc preprocess,
//                                              F kernel,
//                                              dim3 grid_dim,
//                                              dim3 block_dim,
//                                              std::size_t lds_byte,
//                                              float cold_bench_time_secs,
//                                              float bench_hot_time_secs ,
//                                              Args... args)
// {
// #if CK_TIME_KERNEL
//     if(stream_config.time_kernel_)
//     {
//         if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
//         {
//             printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
//                    __func__,
//                    grid_dim.x,
//                    grid_dim.y,
//                    grid_dim.z,
//                    block_dim.x,
//                    block_dim.y,
//                    block_dim.z);

//             // printf("Warm up %d times\n", stream_config.cold_niters_);
//             printf("Cold run time: %.3f seconds\n", cold_bench_time_secs);
//             printf("Hot run time: %.3f seconds\n", hot_bench_time_secs);
//         }
//         // warm up
//         preprocess();
//         for(int i = 0; i < stream_config.cold_niters_; ++i)
//         {
//             kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
//             hip_check_error(hipGetLastError());
//         }

//         const int nrepeat = stream_config.nrepeat_;
//         if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
//         {
//             printf("Start running %d times...\n", nrepeat);
//         }

//         std::vector<hipEvent_t> start_event(2), stop_event(2);
//         for(auto& e : start_event)
//         {
//             hip_check_error(hipEventCreate(&e));
//         }
//         for(auto& e : stop_event)
//         {
//             hip_check_error(hipEventCreate(&e));
//         }
//         hipEvent_t event0;
//         hip_check_error(hipEventCreate(&event0));
//         hip_check_error(hipEventRecord(event0, stream_config.stream_id_));

//         float per_iter_time = 0.f;
//         std::vector<float> times;
//         int i                     = 0;
//         float gpu_time_used       = 0.;
//         // float bench_hot_time_secs     = 0.f; // need to pass as command line argument
//         const float bench_time_ms = bench_hot_time_secs * 1000;

//         while(i < nrepeat || per_iter_time < bench_time_ms)
//         {
//             preprocess();
//             hip_check_error(hipEventRecord(start_event[i % 2], stream_config.stream_id_));
//             kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(gemm_args, args...);
//             hip_check_error(hipEventRecord(stop_event[i % 2], stream_config.stream_id_));

//             if(i > 0)
//             {
//                 // while iteration i is ongoing, wait for iteration i-1 to end
//                 hip_check_error(hipEventSynchronize(stop_event[(i - 1) % 2]));
//                 hip_check_error(hipEventElapsedTime(
//                     &per_iter_time, start_event[(i - 1) % 2], stop_event[(i - 1) % 2]));
//                 // record time for iteration i-1
//                 times.push_back(per_iter_time);
//                 // if iterations 0 to i-1 took more than the required runtime, we can stop
//                 hip_check_error(
//                     hipEventElapsedTime(&per_iter_time, event0, stop_event[(i - 1) % 2]));
//             }
//             i++;
//         }
//         if(!i)
//         {
//             gpu_time_used = 0.;
//         }
//         else
//         {
//             // wait for the final iteration
//             hip_check_error(hipEventSynchronize(stop_event[(i - 1) % 2]));
//             hip_check_error(hipEventElapsedTime(
//                 &per_iter_time, start_event[(i - 1) % 2], stop_event[(i - 1) % 2]));
//             times.push_back(per_iter_time);
//             remove_outliers(times);
//             gpu_time_used = std::accumulate(times.begin(), times.end(), 0.) / times.size();
//             printf("Accumulate time per iteration: %f ms\n",
//                    std::accumulate(times.begin(), times.end(), 0.));
//             printf("total time after removing outliers: %zu \n", times.size());
//             // gpu_time_used *= 1000; // ms to us
//         }
//         return gpu_time_used;
     
//     }
//     else
//     {
//         preprocess();
//         kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
//         hip_check_error(hipGetLastError());

//         return 0;
//     }
// #else
//     kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
//     hip_check_error(hipGetLastError());

//     return 0;
// #endif
// }