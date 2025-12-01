// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <thread>
#include <utility>

namespace ck_tile {

struct joinable_thread : std::thread
{
    template <typename... Xs>
    joinable_thread(Xs&&... xs) : std::thread(std::forward<Xs>(xs)...)
    {
    }

    joinable_thread(joinable_thread&&)            = default;
    joinable_thread& operator=(joinable_thread&&) = default;

    ~joinable_thread()
    {
        if(this->joinable())
            this->join();
    }
};
} // namespace ck_tile
