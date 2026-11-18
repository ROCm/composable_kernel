// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_config.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_schedule_types.hpp"

namespace ck_tile {

enum class FmhaTdmSchedD192Token : index_t
{
    P0M0     = 0,
    P0M1     = 1,
    P0M2     = 2,
    P0M3     = 3,
    P1       = 4,
    P2M0     = 5,
    P2M1     = 6,
    P2M2     = 7,
    P2M3     = 8,
    KM0      = 9,
    KM1      = 10,
    KM2      = 11,
    KM3      = 12,
    VM0      = 13,
    VM1      = 14,
    VM2      = 15,
    VM3      = 16,
    ORescale = 17,
    Tdm      = 18,
    ExpM0    = 19,
    ExpM1    = 20,
    ExpM2    = 21,
    ExpM3    = 22,
};

struct FmhaTdmSchedD192ScheduleRow
{
    static constexpr index_t kMaxTokens = 8;

    FmhaTdmSchedD192Token tokens[kMaxTokens]{};
    index_t size = 0;

    CK_TILE_HOST_DEVICE constexpr FmhaTdmSchedD192Token operator[](index_t i) const
    {
        return tokens[i];
    }
};

struct FmhaTdmSchedGfx125Bf16D192ReadPlacement
{
    using Token            = FmhaTdmSchedD192Token;
    using Row              = FmhaTdmSchedD192ScheduleRow;
    using RequiredGeometry = FmhaTdmSchedBf16D192N128Geometry;
    using Kind             = FmhaTdmSchedLoadKind;

    static constexpr index_t kNumStages       = 4;
    static constexpr index_t kNumQkStages     = kNumStages;
    static constexpr index_t kNumPvStages     = kNumStages;
    static constexpr index_t kQkWmmasPerStage = 24;
    static constexpr index_t kPvWmmasPerStage = 16;
    static constexpr index_t kNumQkRows       = kNumStages * kQkWmmasPerStage;
    static constexpr index_t kNumPvRows       = kNumStages * kPvWmmasPerStage;
    static constexpr index_t kNumWmmaRows     = kNumQkRows + kNumPvRows;

    inline static constexpr Row kQkRows[96] = {
        Row{{Token::Tdm, Token::Tdm, Token::P2M0}, 3},                          // 0:00
        Row{{Token::KM0, Token::KM0, Token::KM0, Token::ORescale}, 4},          // 0:01
        Row{{Token::KM0, Token::KM0, Token::KM0, Token::P2M0, Token::P2M0}, 5}, // 0:02
        Row{{Token::KM1, Token::KM1, Token::KM1, Token::P2M1, Token::P2M1}, 5}, // 0:03
        Row{{Token::KM1, Token::KM1, Token::KM1, Token::P2M1, Token::P2M1}, 5}, // 0:04
        Row{{Token::KM2, Token::KM2, Token::KM2, Token::P2M2, Token::P2M2}, 5}, // 0:05
        Row{{Token::KM2, Token::KM2, Token::KM2, Token::P2M2, Token::P2M0}, 5}, // 0:06
        Row{{Token::KM3, Token::KM3, Token::KM3, Token::P2M3, Token::P2M3}, 5}, // 0:07
        Row{{Token::KM3, Token::KM3, Token::KM3, Token::P2M3}, 4},              // 0:08
        Row{{Token::P2M0, Token::P2M0}, 2},                                     // 0:09
        Row{{Token::P2M0, Token::P2M0}, 2},                                     // 0:10
        Row{{Token::P2M1, Token::P2M1}, 2},                                     // 0:11
        Row{{Token::P2M1, Token::P2M1}, 2},                                     // 0:12
        Row{{Token::P2M1, Token::ORescale}, 2},                                 // 0:13
        Row{{Token::P2M1, Token::P2M0}, 2},                                     // 0:14
        Row{{Token::P2M2, Token::P2M3}, 2},                                     // 0:15
        Row{{Token::P2M2, Token::ORescale}, 2},                                 // 0:16
        Row{{Token::P2M2, Token::ORescale}, 2},                                 // 0:17
        Row{{Token::P2M0, Token::P2M0}, 2},                                     // 0:18
        Row{{Token::P2M0, Token::P2M0}, 2},                                     // 0:19
        Row{{Token::P2M0, Token::P2M0}, 2},                                     // 0:20
        Row{{Token::P2M0, Token::P2M0}, 2},                                     // 0:21
        Row{{Token::P2M0, Token::P2M0}, 2},                                     // 0:22
        Row{{Token::P2M0, Token::P2M2}, 2},                                     // 0:23
        Row{{Token::Tdm, Token::Tdm, Token::P2M2}, 3},                          // 1:00
        Row{{Token::KM0, Token::KM0, Token::KM0, Token::ORescale}, 4},          // 1:01
        Row{{Token::KM0, Token::KM0, Token::KM0, Token::P2M0}, 4},              // 1:02
        Row{{Token::KM1, Token::KM1, Token::KM1, Token::P2M1, Token::P2M1}, 5}, // 1:03
        Row{{Token::KM1, Token::KM1, Token::KM1, Token::P2M1, Token::P2M1}, 5}, // 1:04
        Row{{Token::KM2, Token::KM2, Token::KM2, Token::P2M2, Token::P2M2}, 5}, // 1:05
        Row{{Token::KM2, Token::KM2, Token::KM2, Token::P2M2, Token::P2M2}, 5}, // 1:06
        Row{{Token::KM3, Token::KM3, Token::KM3, Token::P2M3}, 4},              // 1:07
        Row{{Token::KM3, Token::KM3, Token::KM3, Token::P2M3}, 4},              // 1:08
        Row{{Token::P2M3, Token::P2M3}, 2},                                     // 1:09
        Row{{Token::P2M3, Token::P2M3}, 2},                                     // 1:10
        Row{{Token::P2M3, Token::P2M3}, 2},                                     // 1:11
        Row{{Token::P2M1, Token::P2M1}, 2},                                     // 1:12
        Row{{Token::P2M1, Token::P2M1}, 2},                                     // 1:13
        Row{{Token::P2M1, Token::P2M2}, 2},                                     // 1:14
        Row{{Token::P2M1, Token::ORescale}, 2},                                 // 1:15
        Row{{Token::P2M2, Token::ORescale}, 2},                                 // 1:16
        Row{{Token::P2M2, Token::ORescale}, 2},                                 // 1:17
        Row{{Token::P2M2, Token::P2M2}, 2},                                     // 1:18
        Row{{Token::P2M2, Token::P2M2}, 2},                                     // 1:19
        Row{{Token::P2M3, Token::P2M3}, 2},                                     // 1:20
        Row{{Token::P2M3, Token::P2M3}, 2},                                     // 1:21
        Row{{Token::P2M3, Token::P2M3}, 2},                                     // 1:22
        Row{{Token::P2M3, Token::P2M0}, 2},                                     // 1:23
        Row{{Token::KM0, Token::KM0, Token::KM0, Token::P2M0, Token::P2M3}, 5}, // 2:00
        Row{{Token::KM0, Token::KM0, Token::KM0, Token::P2M0, Token::P2M2}, 5}, // 2:01
        Row{{Token::KM1, Token::KM1, Token::KM1, Token::P2M1, Token::P2M1}, 5}, // 2:02
        Row{{Token::KM1, Token::KM1, Token::KM1, Token::P2M1, Token::P2M1}, 5}, // 2:03
        Row{{Token::KM2, Token::KM2, Token::KM2, Token::P2M2, Token::P2M2}, 5}, // 2:04
        Row{{Token::KM2, Token::KM2, Token::KM2, Token::P2M2, Token::P2M2}, 5}, // 2:05
        Row{{Token::KM3, Token::KM3, Token::KM3, Token::P2M3, Token::P2M3}, 5}, // 2:06
        Row{{Token::KM3, Token::KM3, Token::KM3, Token::P2M3, Token::P2M3}, 5}, // 2:07
        Row{{Token::P2M0, Token::P2M0, Token::P2M0, Token::P2M0, Token::P2M0, Token::P2M0},
            6}, // 2:08
        Row{{Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M0},
            7}, // 2:09
        Row{{Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M0},
            7}, // 2:10
        Row{{Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1},
            7}, // 2:11
        Row{{Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1},
            7}, // 2:12
        Row{{Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1},
            7},                                              // 2:13
        Row{{Token::P2M2, Token::P2M2, Token::ORescale}, 3}, // 2:14
        Row{{Token::P2M2, Token::P2M2, Token::ORescale}, 3}, // 2:15
        Row{{Token::P2M2, Token::P2M2, Token::ORescale}, 3}, // 2:16
        Row{{Token::P2M3, Token::P2M3, Token::ORescale}, 3}, // 2:17
        Row{{Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M2},
            6}, // 2:18
        Row{{Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M2},
            6}, // 2:19
        Row{{Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M3, Token::P2M3, Token::P2M3},
            6}, // 2:20
        Row{{Token::P2M3, Token::P2M3, Token::P2M3, Token::P2M3, Token::P2M3, Token::P2M3},
            6}, // 2:21
        Row{{Token::P2M3, Token::P2M3, Token::P2M3, Token::P2M3, Token::P2M3, Token::P2M3},
            6}, // 2:22
        Row{{Token::P2M0, Token::P2M0, Token::P2M0, Token::P2M0, Token::P2M3, Token::P2M3},
            6},                                                                  // 2:23
        Row{{Token::VM0, Token::VM0, Token::P2M0, Token::P2M0}, 4},              // 3:00
        Row{{Token::VM0, Token::VM0, Token::P2M0, Token::P2M0}, 4},              // 3:01
        Row{{Token::VM1, Token::VM1, Token::P2M1, Token::P2M1}, 4},              // 3:02
        Row{{Token::VM1, Token::VM1, Token::P2M1, Token::P2M1, Token::P2M2}, 5}, // 3:03
        Row{{Token::VM2, Token::VM2, Token::P2M2, Token::P2M2}, 4},              // 3:04
        Row{{Token::VM2, Token::VM2, Token::P2M2, Token::P2M2}, 4},              // 3:05
        Row{{Token::VM3, Token::VM3, Token::P2M3, Token::P2M3}, 4},              // 3:06
        Row{{Token::VM3, Token::VM3, Token::P2M3, Token::P2M1}, 4},              // 3:07
        Row{{Token::P2M1, Token::P2M1, Token::P2M3}, 3},                         // 3:08
        Row{{Token::P2M1, Token::P2M0}, 2},                                      // 3:09
        Row{{Token::P2M1, Token::P2M3}, 2},                                      // 3:10
        Row{{Token::P2M1, Token::P2M0, Token::P2M3}, 3},                         // 3:11
        Row{{Token::P2M2, Token::P2M3}, 2},                                      // 3:12
        Row{{Token::P2M2, Token::P2M2}, 2},                                      // 3:13
        Row{{Token::P2M3, Token::P2M3}, 2},                                      // 3:14
        Row{{Token::P2M3, Token::P2M3, Token::P2M1}, 3},                         // 3:15
        Row{{Token::ORescale, Token::P2M2}, 2},                                  // 3:16
        Row{{Token::ORescale, Token::P2M2}, 2},                                  // 3:17
        Row{{Token::ORescale, Token::P2M2, Token::P2M0}, 3},                     // 3:18
        Row{{Token::ORescale, Token::P2M2, Token::P2M3}, 3},                     // 3:19
        Row{{Token::P2M0, Token::P2M3, Token::P2M1}, 3},                         // 3:20
        Row{{Token::P2M0}, 1},                                                   // 3:21
        Row{{Token::P2M3}, 1},                                                   // 3:22
        Row{{}, 0},                                                              // 3:23
    };

    inline static constexpr Row kPvRows[64] = {
        Row{{Token::Tdm, Token::Tdm, Token::P0M0, Token::P0M0, Token::P0M0}, 5}, // 0:00
        Row{{Token::VM0, Token::VM0, Token::P0M0, Token::P0M0, Token::P0M0}, 5}, // 0:01
        Row{{Token::VM0, Token::VM0, Token::P0M0, Token::P0M0, Token::P0M0}, 5}, // 0:02
        Row{{Token::VM1, Token::VM1, Token::P0M1, Token::P0M1, Token::P0M1, Token::P0M1},
            6}, // 0:03
        Row{{Token::VM1, Token::VM1, Token::P0M1, Token::P0M1, Token::P0M1, Token::P0M1},
            6}, // 0:04
        Row{{Token::VM2, Token::VM2, Token::P0M2, Token::P0M2, Token::P0M2, Token::P0M2},
            6}, // 0:05
        Row{{Token::VM2, Token::VM2, Token::P0M2, Token::P0M2, Token::P0M2, Token::P0M2},
            6}, // 0:06
        Row{{Token::VM3, Token::VM3, Token::P0M3, Token::P0M3, Token::P0M3, Token::P0M3},
            6}, // 0:07
        Row{{Token::VM3, Token::VM3, Token::P0M3, Token::P0M3, Token::P0M3, Token::P0M3},
            6},                                                                    // 0:08
        Row{{Token::P0M0, Token::P0M2, Token::P0M2, Token::P0M2, Token::P0M3}, 5}, // 0:09
        Row{{Token::P0M0, Token::P0M1, Token::P0M1, Token::P0M3}, 4},              // 0:10
        Row{{Token::P0M0, Token::P0M0, Token::P0M0, Token::P0M0, Token::P0M1}, 5}, // 0:11
        Row{{Token::P0M0, Token::P0M1, Token::P0M1, Token::P0M2, Token::P0M2}, 5}, // 0:12
        Row{{Token::P0M1, Token::P0M2, Token::P0M2, Token::P0M3, Token::P0M3}, 5}, // 0:13
        Row{{Token::P0M2, Token::P0M3, Token::P0M3, Token::P0M1, Token::P0M1}, 5}, // 0:14
        Row{{Token::P0M3, Token::P0M3}, 2},                                        // 0:15
        Row{{Token::Tdm, Token::Tdm, Token::P0M0}, 3},                             // 1:00
        Row{{Token::VM0, Token::VM0, Token::P0M0, Token::P0M1, Token::P0M2, Token::P0M3},
            6}, // 1:01
        Row{{Token::VM0, Token::VM0, Token::P0M0, Token::P0M1, Token::P0M2, Token::P0M3},
            6}, // 1:02
        Row{{Token::VM1, Token::VM1, Token::P0M1, Token::P0M0, Token::P0M2, Token::P0M3},
            6},                                                                  // 1:03
        Row{{Token::VM1, Token::VM1, Token::P0M1, Token::P0M0}, 4},              // 1:04
        Row{{Token::VM2, Token::VM2, Token::P0M2, Token::P0M2, Token::P0M1}, 5}, // 1:05
        Row{{Token::VM2, Token::VM2, Token::P0M2, Token::P0M0, Token::P0M1}, 5}, // 1:06
        Row{{Token::VM3, Token::VM3, Token::P0M3}, 3},                           // 1:07
        Row{{Token::VM3, Token::VM3, Token::P0M3, Token::P0M3}, 4},              // 1:08
        Row{{Token::P1, Token::P1, Token::P1, Token::P1}, 4},                    // 1:09
        Row{{Token::P1, Token::P1, Token::P1, Token::P1}, 4},                    // 1:10
        Row{{Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M1,
             Token::P2M2,
             Token::P2M2,
             Token::P2M2,
             Token::P2M2},
            8}, // 1:11
        Row{{Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M0,
             Token::P2M3,
             Token::P2M3,
             Token::P2M3,
             Token::P2M3},
            8},                                                       // 1:12
        Row{{Token::P2M0, Token::P2M1, Token::P2M2, Token::P2M3}, 4}, // 1:13
        Row{{Token::P2M0,
             Token::P2M1,
             Token::P2M2,
             Token::P2M3,
             Token::P2M0,
             Token::P2M1,
             Token::P2M2,
             Token::P2M3},
            8},                                                                  // 1:14
        Row{{Token::P2M0, Token::P2M1, Token::P2M2, Token::P2M3}, 4},            // 1:15
        Row{{Token::VM0, Token::VM0, Token::P2M0, Token::P2M0, Token::P2M0}, 5}, // 2:00
        Row{{Token::VM0, Token::VM0, Token::P2M0, Token::P2M0, Token::P2M0}, 5}, // 2:01
        Row{{Token::VM1, Token::VM1, Token::P2M1, Token::P2M2, Token::P2M2}, 5}, // 2:02
        Row{{Token::VM1, Token::VM1, Token::P2M1, Token::P2M2, Token::P2M2}, 5}, // 2:03
        Row{{Token::VM2, Token::VM2, Token::P2M2, Token::P2M1, Token::P2M1}, 5}, // 2:04
        Row{{Token::VM2, Token::VM2, Token::P2M2, Token::P2M1, Token::P2M1}, 5}, // 2:05
        Row{{Token::VM3, Token::VM3, Token::P2M3, Token::P2M3, Token::P2M3, Token::P2M3},
            6},                                                                  // 2:06
        Row{{Token::VM3, Token::VM3, Token::P2M3, Token::P2M3, Token::P2M3}, 5}, // 2:07
        Row{{Token::P2M0, Token::P2M0, Token::P2M0, Token::P2M0, Token::P2M0, Token::P2M0},
            6},                                                       // 2:08
        Row{{Token::P2M0, Token::P2M0, Token::P2M0, Token::P2M1}, 4}, // 2:09
        Row{{Token::P2M1, Token::P2M1, Token::P2M1, Token::P2M1, Token::P2M1, Token::P2M1},
            6},                                                                    // 2:10
        Row{{Token::P2M1, Token::P2M1, Token::P2M1, Token::P2M2, Token::P2M2}, 5}, // 2:11
        Row{{Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M2, Token::P2M2},
            6},                                                                   // 2:12
        Row{{Token::P2M3, Token::P2M3, Token::P2M3}, 3},                          // 2:13
        Row{{Token::P2M3, Token::P2M3, Token::P2M3, Token::P2M3}, 4},             // 2:14
        Row{{Token::P2M0, Token::P2M0, Token::P2M1, Token::P2M1}, 4},             // 2:15
        Row{{Token::KM0, Token::KM0, Token::KM0, Token::ExpM0, Token::ExpM0}, 5}, // 3:00
        Row{{Token::KM0, Token::KM0, Token::KM0, Token::ExpM0}, 4},               // 3:01
        Row{{Token::KM1, Token::KM1, Token::KM1, Token::ExpM1, Token::ExpM1}, 5}, // 3:02
        Row{{Token::KM1, Token::KM1, Token::KM1, Token::ExpM1, Token::ExpM1}, 5}, // 3:03
        Row{{Token::KM2, Token::KM2, Token::KM2, Token::P2M2, Token::ExpM2}, 5},  // 3:04
        Row{{Token::KM2, Token::KM2, Token::KM2, Token::P2M2, Token::ExpM2}, 5},  // 3:05
        Row{{Token::KM3, Token::KM3, Token::KM3, Token::P2M3, Token::ExpM3}, 5},  // 3:06
        Row{{Token::KM3, Token::KM3, Token::KM3, Token::P2M3, Token::ExpM3}, 5},  // 3:07
        Row{{Token::ExpM0, Token::ExpM0, Token::ExpM0}, 3},                       // 3:08
        Row{{Token::ExpM0, Token::ExpM0, Token::ExpM1}, 3},                       // 3:09
        Row{{Token::ExpM1, Token::ExpM1, Token::ExpM1}, 3},                       // 3:10
        Row{{Token::ExpM3, Token::ExpM3, Token::ExpM3}, 3},                       // 3:11
        Row{{Token::ExpM2, Token::ExpM2, Token::ExpM2}, 3},                       // 3:12
        Row{{Token::ExpM2, Token::ExpM2, Token::ExpM2}, 3},                       // 3:13
        Row{{Token::ExpM3, Token::ExpM3, Token::ExpM3}, 3},                       // 3:14
        Row{{}, 0},                                                               // 3:15
    };

    template <index_t N>
    CK_TILE_HOST_DEVICE static constexpr index_t
    CountToken(const Row (&rows)[N], Token token, index_t begin = 0, index_t end = N)
    {
        index_t count = 0;
        for(index_t row = begin; row < end; ++row)
        {
            for(index_t i = 0; i < rows[row].size; ++i)
            {
                count += rows[row][i] == token ? 1 : 0;
            }
        }
        return count;
    }

    template <index_t N>
    CK_TILE_HOST_DEVICE static constexpr index_t
    CountTokenRange(const Row (&rows)[N], Token first, Token last)
    {
        index_t count = 0;
        for(index_t row = 0; row < N; ++row)
        {
            for(index_t i = 0; i < rows[row].size; ++i)
            {
                count += first <= rows[row][i] && rows[row][i] <= last ? 1 : 0;
            }
        }
        return count;
    }

    template <index_t N>
    CK_TILE_HOST_DEVICE static constexpr index_t
    CountTokenBefore(const Row (&rows)[N], Token token, index_t row, index_t slot)
    {
        index_t count = CountToken(rows, token, 0, row);
        for(index_t i = 0; i < slot; ++i)
        {
            count += rows[row][i] == token ? 1 : 0;
        }
        return count;
    }

    template <index_t N>
    CK_TILE_HOST_DEVICE static constexpr index_t FirstTokenPosition(const Row (&rows)[N],
                                                                    Token token)
    {
        for(index_t row = 0; row < N; ++row)
        {
            for(index_t i = 0; i < rows[row].size; ++i)
            {
                if(rows[row][i] == token)
                {
                    return row * Row::kMaxTokens + i;
                }
            }
        }
        return -1;
    }

    template <index_t N>
    CK_TILE_HOST_DEVICE static constexpr index_t LastTokenPosition(const Row (&rows)[N],
                                                                   Token token)
    {
        index_t position = -1;
        for(index_t row = 0; row < N; ++row)
        {
            for(index_t i = 0; i < rows[row].size; ++i)
            {
                if(rows[row][i] == token)
                {
                    position = row * Row::kMaxTokens + i;
                }
            }
        }
        return position;
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr const Row& GetQkRow(index_t wmma)
    {
        static_assert(0 <= Stage && Stage < kNumStages);
        return kQkRows[Stage * kQkWmmasPerStage + wmma];
    }

    template <index_t Stage>
    CK_TILE_HOST_DEVICE static constexpr const Row& GetPvRow(index_t wmma)
    {
        static_assert(0 <= Stage && Stage < kNumStages);
        return kPvRows[Stage * kPvWmmasPerStage + wmma];
    }

    template <index_t Stage, index_t Wmma, index_t Slot>
    CK_TILE_HOST_DEVICE static constexpr index_t GetQkTokenOrdinal()
    {
        static_assert(0 <= Stage && Stage < kNumStages);
        static_assert(0 <= Wmma && Wmma < kQkWmmasPerStage);
        constexpr index_t row = Stage * kQkWmmasPerStage + Wmma;
        static_assert(0 <= Slot && Slot < kQkRows[row].size);
        return CountTokenBefore(kQkRows, kQkRows[row][Slot], row, Slot);
    }

    template <index_t Stage, index_t Wmma, index_t Slot>
    CK_TILE_HOST_DEVICE static constexpr index_t GetPvTokenOrdinal()
    {
        static_assert(0 <= Stage && Stage < kNumStages);
        static_assert(0 <= Wmma && Wmma < kPvWmmasPerStage);
        constexpr index_t row = Stage * kPvWmmasPerStage + Wmma;
        static_assert(0 <= Slot && Slot < kPvRows[row].size);
        return CountTokenBefore(kPvRows, kPvRows[row][Slot], row, Slot);
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitQkRowHalf(Visitor&& visitor)
    {
        constexpr index_t row_index = Stage * kQkWmmasPerStage + Wmma;
        constexpr Row row           = kQkRows[row_index];
        constexpr index_t half      = row.size / 2;

        static_for<0, Row::kMaxTokens, 1>{}([&](auto slot) {
            constexpr bool in_half = SecondHalf ? slot >= half && slot < row.size : slot < half;
            if constexpr(in_half)
            {
                constexpr Token token     = row[slot];
                constexpr index_t ordinal = GetQkTokenOrdinal<Stage, Wmma, slot>();
                visitor(std::integral_constant<Token, token>{}, number<ordinal>{});
            }
        });
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvRowHalf(Visitor&& visitor)
    {
        constexpr index_t row_index = Stage * kPvWmmasPerStage + Wmma;
        constexpr Row row           = kPvRows[row_index];
        constexpr index_t half      = row.size / 2;

        static_for<0, Row::kMaxTokens, 1>{}([&](auto slot) {
            constexpr bool in_half = SecondHalf ? slot >= half && slot < row.size : slot < half;
            if constexpr(in_half)
            {
                constexpr Token token     = row[slot];
                constexpr index_t ordinal = GetPvTokenOrdinal<Stage, Wmma, slot>();
                visitor(std::integral_constant<Token, token>{}, number<ordinal>{});
            }
        });
    }

    template <bool IsQk, index_t Stage, typename TokenConstant, typename Ordinal, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void
    TranslateRead(TokenConstant, Ordinal, Visitor& visitor)
    {
        constexpr auto token = TokenConstant::value;
        if constexpr(token >= Token::KM0 && token <= Token::KM3)
        {
            static_assert(IsQk ? Stage < kNumQkStages - 1 : Stage == kNumPvStages - 1);
            constexpr index_t msb = static_cast<index_t>(token) - static_cast<index_t>(Token::KM0);
            constexpr index_t per_msb = RequiredGeometry::kKSuLoadCount / 4;
            constexpr index_t local   = Ordinal::value - (IsQk ? Stage * per_msb : 0);
            static_assert(local >= 0 && local < per_msb);
            visitor(std::integral_constant<Kind, Kind::KRead>{}, number<msb * per_msb + local>{});
        }
        else if constexpr(token >= Token::VM0 && token <= Token::VM3)
        {
            static_assert(IsQk ? Stage == kNumQkStages - 1 : Stage < kNumPvStages - 1);
            constexpr index_t msb = static_cast<index_t>(token) - static_cast<index_t>(Token::VM0);
            constexpr index_t per_msb = RequiredGeometry::kVStageLoadCount / 4;
            constexpr index_t local   = Ordinal::value - (IsQk ? 0 : Stage * per_msb);
            static_assert(local >= 0 && local < per_msb);
            visitor(std::integral_constant<Kind, Kind::VRead>{}, number<msb * per_msb + local>{});
        }
        else
        {
            // Historical arithmetic names determine the midpoint fence side,
            // but are not operations in the active read placement.
        }
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitQkReadRowHalf(Visitor&& visitor)
    {
        auto translate = [&](auto token, auto ordinal) {
            TranslateRead<true, Stage>(token, ordinal, visitor);
        };
        VisitQkRowHalf<Stage, Wmma, SecondHalf>(translate);
    }

    template <index_t Stage, index_t Wmma, bool SecondHalf, typename Visitor>
    CK_TILE_HOST_DEVICE static constexpr void VisitPvReadRowHalf(Visitor&& visitor)
    {
        auto translate = [&](auto token, auto ordinal) {
            TranslateRead<false, Stage>(token, ordinal, visitor);
        };
        VisitPvRowHalf<Stage, Wmma, SecondHalf>(translate);
    }

    template <index_t N>
    CK_TILE_HOST_DEVICE static constexpr bool ValidateTokenOrdinals(const Row (&rows)[N])
    {
        index_t counts[23] = {};
        for(index_t row = 0; row < N; ++row)
        {
            for(index_t slot = 0; slot < rows[row].size; ++slot)
            {
                const auto token       = rows[row][slot];
                const auto token_index = static_cast<index_t>(token);
                if(CountTokenBefore(rows, token, row, slot) != counts[token_index])
                {
                    return false;
                }
                ++counts[token_index];
            }
        }

        for(index_t token = 0; token < 23; ++token)
        {
            if(counts[token] != CountToken(rows, static_cast<Token>(token)))
            {
                return false;
            }
        }
        return true;
    }

    CK_TILE_HOST_DEVICE static constexpr bool ValidateContract()
    {
        const bool dimensions = kNumQkRows == 96 && kNumPvRows == 64 && kNumWmmaRows == 160;
        const bool qk_counts  = CountTokenRange(kQkRows, Token::KM0, Token::KM3) == 72 &&
                               CountTokenRange(kQkRows, Token::VM0, Token::VM3) == 16 &&
                               CountToken(kQkRows, Token::Tdm) == 4 &&
                               CountTokenRange(kQkRows, Token::P2M0, Token::P2M3) == 228 &&
                               CountToken(kQkRows, Token::ORescale) == 16;
        const bool pv_counts = CountTokenRange(kPvRows, Token::KM0, Token::KM3) == 24 &&
                               CountTokenRange(kPvRows, Token::VM0, Token::VM3) == 48 &&
                               CountToken(kPvRows, Token::Tdm) == 4 &&
                               CountTokenRange(kPvRows, Token::P0M0, Token::P0M3) == 88 &&
                               CountToken(kPvRows, Token::P1) == 8 &&
                               CountTokenRange(kPvRows, Token::P2M0, Token::P2M3) == 99 &&
                               CountTokenRange(kPvRows, Token::ExpM0, Token::ExpM3) == 32;
        const bool total_loads =
            CountTokenRange(kQkRows, Token::KM0, Token::KM3) +
                    CountTokenRange(kPvRows, Token::KM0, Token::KM3) ==
                96 &&
            CountTokenRange(kQkRows, Token::VM0, Token::VM3) +
                    CountTokenRange(kPvRows, Token::VM0, Token::VM3) ==
                64 &&
            CountToken(kQkRows, Token::Tdm) + CountToken(kPvRows, Token::Tdm) == 8;
        const bool stage_tdm = CountToken(kQkRows, Token::Tdm, 0, 24) == 2 &&
                               CountToken(kQkRows, Token::Tdm, 24, 48) == 2 &&
                               CountToken(kQkRows, Token::Tdm, 48, 96) == 0 &&
                               CountToken(kPvRows, Token::Tdm, 0, 16) == 2 &&
                               CountToken(kPvRows, Token::Tdm, 16, 32) == 2 &&
                               CountToken(kPvRows, Token::Tdm, 32, 64) == 0;
        const bool softmax_order =
            LastTokenPosition(kPvRows, Token::P0M0) < FirstTokenPosition(kPvRows, Token::P1) &&
            LastTokenPosition(kPvRows, Token::P0M1) < FirstTokenPosition(kPvRows, Token::P1) &&
            LastTokenPosition(kPvRows, Token::P0M2) < FirstTokenPosition(kPvRows, Token::P1) &&
            LastTokenPosition(kPvRows, Token::P0M3) < FirstTokenPosition(kPvRows, Token::P1) &&
            LastTokenPosition(kPvRows, Token::P1) < FirstTokenPosition(kPvRows, Token::P2M0) &&
            LastTokenPosition(kPvRows, Token::P1) < FirstTokenPosition(kPvRows, Token::P2M1) &&
            LastTokenPosition(kPvRows, Token::P1) < FirstTokenPosition(kPvRows, Token::P2M2) &&
            LastTokenPosition(kPvRows, Token::P1) < FirstTokenPosition(kPvRows, Token::P2M3) &&
            LastTokenPosition(kPvRows, Token::P1) < FirstTokenPosition(kPvRows, Token::ExpM0) &&
            LastTokenPosition(kPvRows, Token::P1) < FirstTokenPosition(kPvRows, Token::ExpM1) &&
            LastTokenPosition(kPvRows, Token::P1) < FirstTokenPosition(kPvRows, Token::ExpM2) &&
            LastTokenPosition(kPvRows, Token::P1) < FirstTokenPosition(kPvRows, Token::ExpM3);
        const bool ordinals = ValidateTokenOrdinals(kQkRows) && ValidateTokenOrdinals(kPvRows);
        return dimensions && qk_counts && pv_counts && total_loads && stage_tdm && softmax_order &&
               ordinals;
    }
};

using FmhaTdmSchedD192Schedule = FmhaTdmSchedGfx125Bf16D192ReadPlacement;

static_assert(FmhaTdmSchedGfx125Bf16D192ReadPlacement::ValidateContract());

} // namespace ck_tile
