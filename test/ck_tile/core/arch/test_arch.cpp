// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "ck_tile/core/arch/arch.hpp"

namespace ck_tile {
namespace test {

// Test amdgcn_target_arch_id values
TEST(TestArch, TargetArchIdValues)
{
    EXPECT_EQ(amdgcn_target_arch_id::GFX908, 0x0908u);
    EXPECT_EQ(amdgcn_target_arch_id::GFX90A, 0x090Au);
    EXPECT_EQ(amdgcn_target_arch_id::GFX942, 0x0942u);
    EXPECT_EQ(amdgcn_target_arch_id::GFX950, 0x0950u);
    EXPECT_EQ(amdgcn_target_arch_id::GFX1100, 0x1100u);
    EXPECT_EQ(amdgcn_target_arch_id::GFX1101, 0x1101u);
    EXPECT_EQ(amdgcn_target_arch_id::GFX1102, 0x1102u);
    EXPECT_EQ(amdgcn_target_arch_id::GFX1151, 0x1151u);
    EXPECT_EQ(amdgcn_target_arch_id::GFX1200, 0x1200u);
    EXPECT_EQ(amdgcn_target_arch_id::GFX1201, 0x1201u);
    EXPECT_EQ(amdgcn_target_arch_id::HOST, 0x0000u);
}

// Test amdgcn_wave_size values
TEST(TestArch, WaveSizeValues)
{
    EXPECT_EQ(amdgcn_wave_size::WAVE32, 32u);
    EXPECT_EQ(amdgcn_wave_size::WAVE64, 64u);
    EXPECT_EQ(amdgcn_wave_size::HOST, 1u);
}

// Test address_space_enum string conversion
TEST(TestArch, AddressSpaceToString)
{
    EXPECT_STREQ(address_space_to_string(address_space_enum::generic), "generic");
    EXPECT_STREQ(address_space_to_string(address_space_enum::global), "global");
    EXPECT_STREQ(address_space_to_string(address_space_enum::lds), "lds");
    EXPECT_STREQ(address_space_to_string(address_space_enum::sgpr), "sgpr");
    EXPECT_STREQ(address_space_to_string(address_space_enum::constant), "constant");
    EXPECT_STREQ(address_space_to_string(address_space_enum::vgpr), "vgpr");
    EXPECT_STREQ(address_space_to_string(static_cast<address_space_enum>(999)), "unknown");
}

// SFINAE test struct for cdna arch id
template <uint32_t GfxTargetId, typename = void>
struct EnableIfCdnaArchIdTest
{
    static bool enabled() { return false; }
};
template <uint32_t GfxTargetId>
struct EnableIfCdnaArchIdTest<GfxTargetId, std::enable_if_t<is_cdna_arch_id<GfxTargetId>::value>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for rdna arch id
template <uint32_t GfxTargetId, typename = void>
struct EnableIfRdnaArchIdTest
{
    static bool enabled() { return false; }
};
template <uint32_t GfxTargetId>
struct EnableIfRdnaArchIdTest<GfxTargetId, std::enable_if_t<is_rdna_arch_id<GfxTargetId>::value>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for gfx9 arch id
template <uint32_t GfxTargetId, typename = void>
struct EnableIfGfx9ArchIdTest
{
    static bool enabled() { return false; }
};
template <uint32_t GfxTargetId>
struct EnableIfGfx9ArchIdTest<GfxTargetId, std::enable_if_t<is_gfx9_arch_id<GfxTargetId>::value>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for gfx11 arch id
template <uint32_t GfxTargetId, typename = void>
struct EnableIfGfx11ArchIdTest
{
    static bool enabled() { return false; }
};
template <uint32_t GfxTargetId>
struct EnableIfGfx11ArchIdTest<GfxTargetId, std::enable_if_t<is_gfx11_arch_id<GfxTargetId>::value>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for gfx12 arch id
template <uint32_t GfxTargetId, typename = void>
struct EnableIfGfx12ArchIdTest
{
    static bool enabled() { return false; }
};
template <uint32_t GfxTargetId>
struct EnableIfGfx12ArchIdTest<GfxTargetId, std::enable_if_t<is_gfx12_arch_id<GfxTargetId>::value>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for wave32 arch id
template <uint32_t GfxTargetId, typename = void>
struct EnableIfWave32ArchIdTest
{
    static bool enabled() { return false; }
};
template <uint32_t GfxTargetId>
struct EnableIfWave32ArchIdTest<GfxTargetId,
                                std::enable_if_t<is_wave32_arch_id<GfxTargetId>::value>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for wave64 arch id
template <uint32_t GfxTargetId, typename = void>
struct EnableIfWave64ArchIdTest
{
    static bool enabled() { return false; }
};
template <uint32_t GfxTargetId>
struct EnableIfWave64ArchIdTest<GfxTargetId,
                                std::enable_if_t<is_wave64_arch_id<GfxTargetId>::value>>
{
    static bool enabled() { return true; }
};

// Additional tests for all amdgcn_target_arch_id values
TEST(TestArch, IsCdnaArchId_AllArchIds)
{
    EXPECT_TRUE((is_cdna_arch_id<amdgcn_target_arch_id::GFX908>::value));
    EXPECT_TRUE((is_cdna_arch_id<amdgcn_target_arch_id::GFX90A>::value));
    EXPECT_TRUE((is_cdna_arch_id<amdgcn_target_arch_id::GFX942>::value));
    EXPECT_TRUE((is_cdna_arch_id<amdgcn_target_arch_id::GFX950>::value));
    EXPECT_FALSE((is_cdna_arch_id<amdgcn_target_arch_id::GFX1100>::value));
    EXPECT_FALSE((is_cdna_arch_id<amdgcn_target_arch_id::GFX1101>::value));
    EXPECT_FALSE((is_cdna_arch_id<amdgcn_target_arch_id::GFX1102>::value));
    EXPECT_FALSE((is_cdna_arch_id<amdgcn_target_arch_id::GFX1151>::value));
    EXPECT_FALSE((is_cdna_arch_id<amdgcn_target_arch_id::GFX1200>::value));
    EXPECT_FALSE((is_cdna_arch_id<amdgcn_target_arch_id::GFX1201>::value));
    EXPECT_FALSE((is_cdna_arch_id<amdgcn_target_arch_id::HOST>::value));
}

TEST(TestArch, IsRdnaArchId_AllArchIds)
{
    EXPECT_FALSE((is_rdna_arch_id<amdgcn_target_arch_id::GFX908>::value));
    EXPECT_FALSE((is_rdna_arch_id<amdgcn_target_arch_id::GFX90A>::value));
    EXPECT_FALSE((is_rdna_arch_id<amdgcn_target_arch_id::GFX942>::value));
    EXPECT_FALSE((is_rdna_arch_id<amdgcn_target_arch_id::GFX950>::value));
    EXPECT_TRUE((is_rdna_arch_id<amdgcn_target_arch_id::GFX1100>::value));
    EXPECT_TRUE((is_rdna_arch_id<amdgcn_target_arch_id::GFX1101>::value));
    EXPECT_TRUE((is_rdna_arch_id<amdgcn_target_arch_id::GFX1102>::value));
    EXPECT_TRUE((is_rdna_arch_id<amdgcn_target_arch_id::GFX1151>::value));
    EXPECT_TRUE((is_rdna_arch_id<amdgcn_target_arch_id::GFX1200>::value));
    EXPECT_TRUE((is_rdna_arch_id<amdgcn_target_arch_id::GFX1201>::value));
    EXPECT_FALSE((is_rdna_arch_id<amdgcn_target_arch_id::HOST>::value));
}

TEST(TestArch, IsGfx9ArchId_AllArchIds)
{
    EXPECT_TRUE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX908>::value));
    EXPECT_TRUE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX90A>::value));
    EXPECT_TRUE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX942>::value));
    EXPECT_TRUE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX950>::value));
    EXPECT_FALSE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX1100>::value));
    EXPECT_FALSE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX1101>::value));
    EXPECT_FALSE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX1102>::value));
    EXPECT_FALSE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX1151>::value));
    EXPECT_FALSE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX1200>::value));
    EXPECT_FALSE((is_gfx9_arch_id<amdgcn_target_arch_id::GFX1201>::value));
    EXPECT_FALSE((is_gfx9_arch_id<amdgcn_target_arch_id::HOST>::value));
}

TEST(TestArch, IsGfx11ArchId_AllArchIds)
{
    EXPECT_FALSE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX908>::value));
    EXPECT_FALSE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX90A>::value));
    EXPECT_FALSE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX942>::value));
    EXPECT_FALSE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX950>::value));
    EXPECT_TRUE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX1100>::value));
    EXPECT_TRUE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX1101>::value));
    EXPECT_TRUE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX1102>::value));
    EXPECT_TRUE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX1151>::value));
    EXPECT_FALSE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX1200>::value));
    EXPECT_FALSE((is_gfx11_arch_id<amdgcn_target_arch_id::GFX1201>::value));
    EXPECT_FALSE((is_gfx11_arch_id<amdgcn_target_arch_id::HOST>::value));
}

TEST(TestArch, IsGfx12ArchId_AllArchIds)
{
    EXPECT_FALSE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX908>::value));
    EXPECT_FALSE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX90A>::value));
    EXPECT_FALSE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX942>::value));
    EXPECT_FALSE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX950>::value));
    EXPECT_FALSE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX1100>::value));
    EXPECT_FALSE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX1101>::value));
    EXPECT_FALSE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX1102>::value));
    EXPECT_FALSE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX1151>::value));
    EXPECT_TRUE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX1200>::value));
    EXPECT_TRUE((is_gfx12_arch_id<amdgcn_target_arch_id::GFX1201>::value));
    EXPECT_FALSE((is_gfx12_arch_id<amdgcn_target_arch_id::HOST>::value));
}

TEST(TestArch, IsWave32ArchId_AllArchIds)
{
    EXPECT_FALSE((is_wave32_arch_id<amdgcn_target_arch_id::GFX908>::value));
    EXPECT_FALSE((is_wave32_arch_id<amdgcn_target_arch_id::GFX90A>::value));
    EXPECT_FALSE((is_wave32_arch_id<amdgcn_target_arch_id::GFX942>::value));
    EXPECT_FALSE((is_wave32_arch_id<amdgcn_target_arch_id::GFX950>::value));
    EXPECT_TRUE((is_wave32_arch_id<amdgcn_target_arch_id::GFX1100>::value));
    EXPECT_TRUE((is_wave32_arch_id<amdgcn_target_arch_id::GFX1101>::value));
    EXPECT_TRUE((is_wave32_arch_id<amdgcn_target_arch_id::GFX1102>::value));
    EXPECT_TRUE((is_wave32_arch_id<amdgcn_target_arch_id::GFX1151>::value));
    EXPECT_TRUE((is_wave32_arch_id<amdgcn_target_arch_id::GFX1200>::value));
    EXPECT_TRUE((is_wave32_arch_id<amdgcn_target_arch_id::GFX1201>::value));
    EXPECT_FALSE((is_wave32_arch_id<amdgcn_target_arch_id::HOST>::value));
}

TEST(TestArch, IsWave64ArchId_AllArchIds)
{
    EXPECT_TRUE((is_wave64_arch_id<amdgcn_target_arch_id::GFX908>::value));
    EXPECT_TRUE((is_wave64_arch_id<amdgcn_target_arch_id::GFX90A>::value));
    EXPECT_TRUE((is_wave64_arch_id<amdgcn_target_arch_id::GFX942>::value));
    EXPECT_TRUE((is_wave64_arch_id<amdgcn_target_arch_id::GFX950>::value));
    EXPECT_FALSE((is_wave64_arch_id<amdgcn_target_arch_id::GFX1100>::value));
    EXPECT_FALSE((is_wave64_arch_id<amdgcn_target_arch_id::GFX1101>::value));
    EXPECT_FALSE((is_wave64_arch_id<amdgcn_target_arch_id::GFX1102>::value));
    EXPECT_FALSE((is_wave64_arch_id<amdgcn_target_arch_id::GFX1151>::value));
    EXPECT_FALSE((is_wave64_arch_id<amdgcn_target_arch_id::GFX1200>::value));
    EXPECT_FALSE((is_wave64_arch_id<amdgcn_target_arch_id::GFX1201>::value));
    EXPECT_FALSE((is_wave64_arch_id<amdgcn_target_arch_id::HOST>::value));
}

TEST(TestArch, EnableIfCdnaArchId_AllArchIds)
{
    EXPECT_TRUE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX908>::enabled()));
    EXPECT_TRUE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX90A>::enabled()));
    EXPECT_TRUE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX942>::enabled()));
    EXPECT_TRUE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX950>::enabled()));
    EXPECT_FALSE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX1100>::enabled()));
    EXPECT_FALSE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX1101>::enabled()));
    EXPECT_FALSE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX1102>::enabled()));
    EXPECT_FALSE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX1151>::enabled()));
    EXPECT_FALSE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX1200>::enabled()));
    EXPECT_FALSE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::GFX1201>::enabled()));
    EXPECT_FALSE((EnableIfCdnaArchIdTest<amdgcn_target_arch_id::HOST>::enabled()));
}

TEST(TestArch, EnableIfRdnaArchId_AllArchIds)
{
    EXPECT_FALSE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX908>::enabled()));
    EXPECT_FALSE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX90A>::enabled()));
    EXPECT_FALSE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX942>::enabled()));
    EXPECT_FALSE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX950>::enabled()));
    EXPECT_TRUE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX1100>::enabled()));
    EXPECT_TRUE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX1101>::enabled()));
    EXPECT_TRUE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX1102>::enabled()));
    EXPECT_TRUE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX1151>::enabled()));
    EXPECT_TRUE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX1200>::enabled()));
    EXPECT_TRUE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::GFX1201>::enabled()));
    EXPECT_FALSE((EnableIfRdnaArchIdTest<amdgcn_target_arch_id::HOST>::enabled()));
}

TEST(TestArch, EnableIfGfx9ArchId_AllArchIds)
{
    EXPECT_TRUE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX908>::enabled()));
    EXPECT_TRUE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX90A>::enabled()));
    EXPECT_TRUE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX942>::enabled()));
    EXPECT_TRUE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX950>::enabled()));
    EXPECT_FALSE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX1100>::enabled()));
    EXPECT_FALSE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX1101>::enabled()));
    EXPECT_FALSE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX1102>::enabled()));
    EXPECT_FALSE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX1151>::enabled()));
    EXPECT_FALSE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX1200>::enabled()));
    EXPECT_FALSE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::GFX1201>::enabled()));
    EXPECT_FALSE((EnableIfGfx9ArchIdTest<amdgcn_target_arch_id::HOST>::enabled()));
}

TEST(TestArch, EnableIfGfx11ArchId_AllArchIds)
{
    EXPECT_FALSE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX908>::enabled()));
    EXPECT_FALSE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX90A>::enabled()));
    EXPECT_FALSE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX942>::enabled()));
    EXPECT_FALSE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX950>::enabled()));
    EXPECT_TRUE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX1100>::enabled()));
    EXPECT_TRUE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX1101>::enabled()));
    EXPECT_TRUE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX1102>::enabled()));
    EXPECT_TRUE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX1151>::enabled()));
    EXPECT_FALSE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX1200>::enabled()));
    EXPECT_FALSE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::GFX1201>::enabled()));
    EXPECT_FALSE((EnableIfGfx11ArchIdTest<amdgcn_target_arch_id::HOST>::enabled()));
}

TEST(TestArch, EnableIfGfx12ArchId_AllArchIds)
{
    EXPECT_FALSE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX908>::enabled()));
    EXPECT_FALSE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX90A>::enabled()));
    EXPECT_FALSE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX942>::enabled()));
    EXPECT_FALSE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX950>::enabled()));
    EXPECT_FALSE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX1100>::enabled()));
    EXPECT_FALSE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX1101>::enabled()));
    EXPECT_FALSE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX1102>::enabled()));
    EXPECT_FALSE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX1151>::enabled()));
    EXPECT_TRUE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX1200>::enabled()));
    EXPECT_TRUE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::GFX1201>::enabled()));
    EXPECT_FALSE((EnableIfGfx12ArchIdTest<amdgcn_target_arch_id::HOST>::enabled()));
}

TEST(TestArch, EnableIfWave32ArchId_AllArchIds)
{
    EXPECT_FALSE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX908>::enabled()));
    EXPECT_FALSE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX90A>::enabled()));
    EXPECT_FALSE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX942>::enabled()));
    EXPECT_FALSE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX950>::enabled()));
    EXPECT_TRUE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX1100>::enabled()));
    EXPECT_TRUE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX1101>::enabled()));
    EXPECT_TRUE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX1102>::enabled()));
    EXPECT_TRUE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX1151>::enabled()));
    EXPECT_TRUE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX1200>::enabled()));
    EXPECT_TRUE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::GFX1201>::enabled()));
    EXPECT_FALSE((EnableIfWave32ArchIdTest<amdgcn_target_arch_id::HOST>::enabled()));
}

TEST(TestArch, EnableIfWave64ArchId_AllArchIds)
{
    EXPECT_TRUE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX908>::enabled()));
    EXPECT_TRUE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX90A>::enabled()));
    EXPECT_TRUE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX942>::enabled()));
    EXPECT_TRUE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX950>::enabled()));
    EXPECT_FALSE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX1100>::enabled()));
    EXPECT_FALSE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX1101>::enabled()));
    EXPECT_FALSE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX1102>::enabled()));
    EXPECT_FALSE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX1151>::enabled()));
    EXPECT_FALSE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX1200>::enabled()));
    EXPECT_FALSE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::GFX1201>::enabled()));
    EXPECT_FALSE((EnableIfWave64ArchIdTest<amdgcn_target_arch_id::HOST>::enabled()));
}

} // namespace test
} // namespace ck_tile
