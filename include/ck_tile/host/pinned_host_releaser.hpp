// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <condition_variable>
#include <hip/hip_runtime.h>
#include <mutex>
#include <queue>
#include <thread>

namespace ck_tile {

// Defers hipHostFree off the HIP callback path. HIP callbacks hold runtime
// locks, so calling hipHostFree (or any HIP API) from one deadlocks against
// concurrent main-thread hipFree. enqueue() is HIP-API-free; a worker thread
// drains the queue and calls hipHostFree. Use instance() for a process-wide
// shared worker.
class pinned_host_releaser
{
    std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<void*> q_;
    std::thread worker_;
    bool stop_ = false;

    void run()
    {
        for(;;)
        {
            void* p = nullptr;
            {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait(lk, [&] { return stop_ || !q_.empty(); });
                if(q_.empty())
                    return; // stop_ && empty
                p = q_.front();
                q_.pop();
            }
            (void)hipHostFree(p);
        }
    }

    public:
    pinned_host_releaser() : worker_([this] { run(); }) {}

    ~pinned_host_releaser()
    {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            stop_ = true;
        }
        cv_.notify_all();
        if(worker_.joinable())
            worker_.join();
    }

    pinned_host_releaser(const pinned_host_releaser&)            = delete;
    pinned_host_releaser& operator=(const pinned_host_releaser&) = delete;

    static pinned_host_releaser& instance()
    {
        static pinned_host_releaser r;
        return r;
    }

    void enqueue(void* p)
    {
        {
            std::lock_guard<std::mutex> lk(mtx_);
            q_.push(p);
        }
        cv_.notify_one();
    }
};

} // namespace ck_tile
