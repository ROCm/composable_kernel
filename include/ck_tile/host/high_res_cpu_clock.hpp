// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>
#if defined(_WIN32) || defined(_WIN64)
// Windows
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ck_tile {

// Time structure to hold nanoseconds since epoch or arbitrary start point
struct timepoint_t
{
    int64_t nanoseconds;
};

// Platform-specific includes and implementation
#if defined(_WIN32) || defined(_WIN64)

static inline timepoint_t high_res_now()
{
    // Cache the performance counter frequency; it is constant for the system lifetime.
    static LARGE_INTEGER frequency = []() {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        return f;
    }();

    LARGE_INTEGER counter;
    timepoint_t tp;
    QueryPerformanceCounter(&counter);

    // Convert to nanoseconds using floating-point to avoid 64-bit integer overflow
    tp.nanoseconds =
        static_cast<int64_t>((static_cast<long double>(counter.QuadPart) * 1000000000.0L) /
                             static_cast<long double>(frequency.QuadPart));

    return tp;
}

#elif defined(__linux__) || defined(__unix__) || defined(_POSIX_VERSION)
// Linux/Unix/POSIX
#include <time.h>

static inline timepoint_t high_res_now()
{
    struct timespec ts;
    timepoint_t tp;

    // Use CLOCK_MONOTONIC for consistent timing unaffected by system time changes
    // Use CLOCK_REALTIME if you need wall-clock time
    clock_gettime(CLOCK_MONOTONIC, &ts);

    tp.nanoseconds = static_cast<int64_t>(ts.tv_sec * 1000000000LL + ts.tv_nsec);

    return tp;
}

#else
// Fallback for other platforms
#include <time.h>

static inline timepoint_t high_res_now()
{
    timepoint_t tp;
    time_t t       = time(NULL);
    tp.nanoseconds = static_cast<int64_t>(t * 1000000000LL);
    return tp;
}

#endif

// Duration calculation functions
static inline int64_t duration_ns(timepoint_t start, timepoint_t end)
{
    return end.nanoseconds - start.nanoseconds;
}

static inline int64_t duration_us(timepoint_t start, timepoint_t end)
{
    return (end.nanoseconds - start.nanoseconds) / 1000LL;
}

static inline int64_t duration_ms(timepoint_t start, timepoint_t end)
{
    return (end.nanoseconds - start.nanoseconds) / 1000000LL;
}

static inline double duration_sec(timepoint_t start, timepoint_t end)
{
    return static_cast<double>(end.nanoseconds - start.nanoseconds) / 1000000000.0;
}

} // namespace ck_tile
