// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/dispatcher/fmha_kernel_key.hpp"
#include "ck_tile/dispatcher/fmha_problem.hpp"

#include "ck_tile/host/kernel_launch.hpp"

#include <memory>
#include <string>

namespace ck_tile {
namespace dispatcher {

class FmhaKernelInstance
{
    public:
    virtual ~FmhaKernelInstance() = default;

    [[nodiscard]] virtual const FmhaKernelKey& get_key() const            = 0;
    [[nodiscard]] virtual bool supports(const FmhaProblem& problem) const = 0;
    [[nodiscard]] virtual std::string get_name() const                    = 0;

    // Short aliases (preferred for new code)
    [[nodiscard]] const FmhaKernelKey& key() const { return get_key(); }
    [[nodiscard]] std::string name() const { return get_name(); }

    virtual void launch(const FmhaInvocation& invocation,
                        const ck_tile::stream_config& stream_config) const = 0;

    [[nodiscard]] virtual float run(const FmhaInvocation& invocation,
                                    const ck_tile::stream_config& stream_config) const
    {
        return ck_tile::launch_kernel(
            stream_config,
            [this, &invocation](const ck_tile::stream_config& sc) { launch(invocation, sc); });
    }
};

using FmhaKernelInstancePtr = std::shared_ptr<FmhaKernelInstance>;

} // namespace dispatcher
} // namespace ck_tile
