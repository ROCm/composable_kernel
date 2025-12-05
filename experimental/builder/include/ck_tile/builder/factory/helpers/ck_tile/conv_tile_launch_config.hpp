// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/conv_algorithm_concepts.hpp"

namespace ck_tile::builder::factory::internal {

struct TileLaunchConfig
{
    bool has_hot_loop                               = false;
    ck_tile::TailNumber tail_number                 = ck_tile::TailNumber::Full;
    ck_tile::memory_operation_enum memory_operation = ck_tile::memory_operation_enum::set;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck_tile::TailNumber SetTailNumber()
{
    constexpr auto tail_number = ALGORITHM.launch_config.tail_number;
    using ck_tile_tail_num     = ck_tile::TailNumber;
    switch(tail_number)
    {
    case TailNumber::ODD: return ck_tile_tail_num::Odd;
    case TailNumber::EVEN: return ck_tile_tail_num::Even;
    case TailNumber::ONE: return ck_tile_tail_num::One;
    case TailNumber::TWO: return ck_tile_tail_num::Two;
    case TailNumber::THREE: return ck_tile_tail_num::Three;
    case TailNumber::FOUR: return ck_tile_tail_num::Four;
    case TailNumber::FIVE: return ck_tile_tail_num::Five;
    case TailNumber::SIX: return ck_tile_tail_num::Six;
    case TailNumber::SEVEN: return ck_tile_tail_num::Seven;
    case TailNumber::EMPTY: return ck_tile_tail_num::Empty;
    case TailNumber::FULL: return ck_tile_tail_num::Full;
    default: throw "Unknown Tail Number";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck_tile::memory_operation_enum SetMemoryOperation()
{
    constexpr auto memory_operation = ALGORITHM.launch_config.memory_operation;
    using ck_tile_mem_op            = ck_tile::memory_operation_enum;
    switch(memory_operation)
    {
    case MemoryOperation::SET: return ck_tile_mem_op::set;
    case MemoryOperation::ATOMIC_ADD: return ck_tile_mem_op::atomic_add;
    case MemoryOperation::ATOMIC_MAX: return ck_tile_mem_op::atomic_max;
    case MemoryOperation::ADD: return ck_tile_mem_op::add;
    default: throw "Unknown Memory Operation";
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr TileLaunchConfig SetTileLaunchConfig()
{
    return TileLaunchConfig{.has_hot_loop     = ALGORITHM.launch_config.has_hot_loop,
                            .tail_number      = SetTailNumber<ALGORITHM>(),
                            .memory_operation = SetMemoryOperation<ALGORITHM>()};
}

} // namespace ck_tile::builder::factory::internal
