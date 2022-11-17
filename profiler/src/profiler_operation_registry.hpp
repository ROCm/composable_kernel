// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <functional>
#include <map>
#include <optional>
#include <string_view>

class ProfilerOperationRegistry final
{
    ProfilerOperationRegistry() = default;

    public:
    using Operation = std::function<int(int, char*[])>;

    private:
    std::unordered_map<std::string_view, Operation> operations_;

    public:
    static ProfilerOperationRegistry& GetInstance()
    {
        static ProfilerOperationRegistry registry;
        return registry;
    }

    std::optional<Operation> Get(std::string_view name) const
    {
        const auto found = operations_.find(name);
        if(found == end(operations_))
        {
            return std::nullopt;
        }

        return found->second;
    }

    bool Add(std::string_view name, Operation operation)
    {
        return operations_.try_emplace(name, std::move(operation)).second;
    }
};

#define REGISTER_PROFILER_OPERATION(name, operation)                                     \
    namespace {                                                                          \
    const bool result = ::ProfilerOperationRegistry::GetInstance().Add(name, operation); \
    }
