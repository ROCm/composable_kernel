// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "ck_tile/core/arch/arch.hpp"

namespace ck_tile {

// Test amdgcn_wave_size values
TEST(TestArch, WaveSizeValues)
{
    EXPECT_EQ(static_cast<uint32_t>(amdgcn_wave_size::WAVE32), 32u);
    EXPECT_EQ(static_cast<uint32_t>(amdgcn_wave_size::WAVE64), 64u);
    EXPECT_EQ(static_cast<uint32_t>(amdgcn_wave_size::HOST), 1u);
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
template <amdgcn_target_arch_id GfxTargetId, typename = void>
struct EnableIfCdnaArchIdTest
{
    static bool enabled() { return false; }
};
template <amdgcn_target_arch_id GfxTargetId>
struct EnableIfCdnaArchIdTest<GfxTargetId, enable_if_cdna_target_id_t<GfxTargetId>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for rdna arch id
template <amdgcn_target_arch_id GfxTargetId, typename = void>
struct EnableIfRdnaArchIdTest
{
    static bool enabled() { return false; }
};
template <amdgcn_target_arch_id GfxTargetId>
struct EnableIfRdnaArchIdTest<GfxTargetId, enable_if_rdna_target_id_t<GfxTargetId>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for gfx9 arch id
template <amdgcn_target_arch_id GfxTargetId, typename = void>
struct EnableIfGfx9ArchIdTest
{
    static bool enabled() { return false; }
};
template <amdgcn_target_arch_id GfxTargetId>
struct EnableIfGfx9ArchIdTest<GfxTargetId, enable_if_gfx9_target_id_t<GfxTargetId>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for gfx11 arch id
template <amdgcn_target_arch_id GfxTargetId, typename = void>
struct EnableIfGfx11ArchIdTest
{
    static bool enabled() { return false; }
};
template <amdgcn_target_arch_id GfxTargetId>
struct EnableIfGfx11ArchIdTest<GfxTargetId, enable_if_gfx11_target_id_t<GfxTargetId>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for gfx12 arch id
template <amdgcn_target_arch_id GfxTargetId, typename = void>
struct EnableIfGfx12ArchIdTest
{
    static bool enabled() { return false; }
};
template <amdgcn_target_arch_id GfxTargetId>
struct EnableIfGfx12ArchIdTest<GfxTargetId, enable_if_gfx12_target_id_t<GfxTargetId>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for wave32 arch id
template <amdgcn_target_arch_id GfxTargetId, typename = void>
struct EnableIfWave32ArchIdTest
{
    static bool enabled() { return false; }
};
template <amdgcn_target_arch_id GfxTargetId>
struct EnableIfWave32ArchIdTest<GfxTargetId, enable_if_wave32_target_id_t<GfxTargetId>>
{
    static bool enabled() { return true; }
};

// SFINAE test struct for wave64 arch id
template <amdgcn_target_arch_id GfxTargetId, typename = void>
struct EnableIfWave64ArchIdTest
{
    static bool enabled() { return false; }
};
template <amdgcn_target_arch_id GfxTargetId>
struct EnableIfWave64ArchIdTest<GfxTargetId, enable_if_wave64_target_id_t<GfxTargetId>>
{
    static bool enabled() { return true; }
};

// Additional tests for all amdgcn_target_arch_id values
TEST(TestArch, IsCdnaArchId_AllArchIds)
{
    EXPECT_TRUE(is_cdna_arch_id(amdgcn_target_arch_id::GFX908));
    EXPECT_TRUE(is_cdna_arch_id(amdgcn_target_arch_id::GFX90A));
    EXPECT_TRUE(is_cdna_arch_id(amdgcn_target_arch_id::GFX942));
    EXPECT_TRUE(is_cdna_arch_id(amdgcn_target_arch_id::GFX950));
    EXPECT_FALSE(is_cdna_arch_id(amdgcn_target_arch_id::GFX1100));
    EXPECT_FALSE(is_cdna_arch_id(amdgcn_target_arch_id::GFX1101));
    EXPECT_FALSE(is_cdna_arch_id(amdgcn_target_arch_id::GFX1102));
    EXPECT_FALSE(is_cdna_arch_id(amdgcn_target_arch_id::GFX1151));
    EXPECT_FALSE(is_cdna_arch_id(amdgcn_target_arch_id::GFX1200));
    EXPECT_FALSE(is_cdna_arch_id(amdgcn_target_arch_id::GFX1201));
    EXPECT_FALSE(is_cdna_arch_id(amdgcn_target_arch_id::HOST));
}

TEST(TestArch, IsRdnaArchId_AllArchIds)
{
    EXPECT_FALSE(is_rdna_arch_id(amdgcn_target_arch_id::GFX908));
    EXPECT_FALSE(is_rdna_arch_id(amdgcn_target_arch_id::GFX90A));
    EXPECT_FALSE(is_rdna_arch_id(amdgcn_target_arch_id::GFX942));
    EXPECT_FALSE(is_rdna_arch_id(amdgcn_target_arch_id::GFX950));
    EXPECT_TRUE(is_rdna_arch_id(amdgcn_target_arch_id::GFX1100));
    EXPECT_TRUE(is_rdna_arch_id(amdgcn_target_arch_id::GFX1101));
    EXPECT_TRUE(is_rdna_arch_id(amdgcn_target_arch_id::GFX1102));
    EXPECT_TRUE(is_rdna_arch_id(amdgcn_target_arch_id::GFX1151));
    EXPECT_TRUE(is_rdna_arch_id(amdgcn_target_arch_id::GFX1200));
    EXPECT_TRUE(is_rdna_arch_id(amdgcn_target_arch_id::GFX1201));
    EXPECT_FALSE(is_rdna_arch_id(amdgcn_target_arch_id::HOST));
}

TEST(TestArch, IsGfx9ArchId_AllArchIds)
{
    EXPECT_TRUE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX908));
    EXPECT_TRUE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX90A));
    EXPECT_TRUE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX942));
    EXPECT_TRUE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX950));
    EXPECT_FALSE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX1100));
    EXPECT_FALSE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX1101));
    EXPECT_FALSE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX1102));
    EXPECT_FALSE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX1151));
    EXPECT_FALSE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX1200));
    EXPECT_FALSE(is_gfx9_arch_id(amdgcn_target_arch_id::GFX1201));
    EXPECT_FALSE(is_gfx9_arch_id(amdgcn_target_arch_id::HOST));
}

TEST(TestArch, IsGfx11ArchId_AllArchIds)
{
    EXPECT_FALSE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX908));
    EXPECT_FALSE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX90A));
    EXPECT_FALSE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX942));
    EXPECT_FALSE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX950));
    EXPECT_TRUE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX1100));
    EXPECT_TRUE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX1101));
    EXPECT_TRUE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX1102));
    EXPECT_TRUE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX1151));
    EXPECT_FALSE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX1200));
    EXPECT_FALSE(is_gfx11_arch_id(amdgcn_target_arch_id::GFX1201));
    EXPECT_FALSE(is_gfx11_arch_id(amdgcn_target_arch_id::HOST));
}

TEST(TestArch, IsGfx12ArchId_AllArchIds)
{
    EXPECT_FALSE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX908));
    EXPECT_FALSE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX90A));
    EXPECT_FALSE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX942));
    EXPECT_FALSE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX950));
    EXPECT_FALSE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX1100));
    EXPECT_FALSE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX1101));
    EXPECT_FALSE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX1102));
    EXPECT_FALSE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX1151));
    EXPECT_TRUE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX1200));
    EXPECT_TRUE(is_gfx12_arch_id(amdgcn_target_arch_id::GFX1201));
    EXPECT_FALSE(is_gfx12_arch_id(amdgcn_target_arch_id::HOST));
}

TEST(TestArch, IsWave32ArchId_AllArchIds)
{
    EXPECT_FALSE(is_wave32_arch_id(amdgcn_target_arch_id::GFX908));
    EXPECT_FALSE(is_wave32_arch_id(amdgcn_target_arch_id::GFX90A));
    EXPECT_FALSE(is_wave32_arch_id(amdgcn_target_arch_id::GFX942));
    EXPECT_FALSE(is_wave32_arch_id(amdgcn_target_arch_id::GFX950));
    EXPECT_TRUE(is_wave32_arch_id(amdgcn_target_arch_id::GFX1100));
    EXPECT_TRUE(is_wave32_arch_id(amdgcn_target_arch_id::GFX1101));
    EXPECT_TRUE(is_wave32_arch_id(amdgcn_target_arch_id::GFX1102));
    EXPECT_TRUE(is_wave32_arch_id(amdgcn_target_arch_id::GFX1151));
    EXPECT_TRUE(is_wave32_arch_id(amdgcn_target_arch_id::GFX1200));
    EXPECT_TRUE(is_wave32_arch_id(amdgcn_target_arch_id::GFX1201));
    EXPECT_FALSE(is_wave32_arch_id(amdgcn_target_arch_id::HOST));
}

TEST(TestArch, IsWave64ArchId_AllArchIds)
{
    EXPECT_TRUE(is_wave64_arch_id(amdgcn_target_arch_id::GFX908));
    EXPECT_TRUE(is_wave64_arch_id(amdgcn_target_arch_id::GFX90A));
    EXPECT_TRUE(is_wave64_arch_id(amdgcn_target_arch_id::GFX942));
    EXPECT_TRUE(is_wave64_arch_id(amdgcn_target_arch_id::GFX950));
    EXPECT_FALSE(is_wave64_arch_id(amdgcn_target_arch_id::GFX1100));
    EXPECT_FALSE(is_wave64_arch_id(amdgcn_target_arch_id::GFX1101));
    EXPECT_FALSE(is_wave64_arch_id(amdgcn_target_arch_id::GFX1102));
    EXPECT_FALSE(is_wave64_arch_id(amdgcn_target_arch_id::GFX1151));
    EXPECT_FALSE(is_wave64_arch_id(amdgcn_target_arch_id::GFX1200));
    EXPECT_FALSE(is_wave64_arch_id(amdgcn_target_arch_id::GFX1201));
    EXPECT_FALSE(is_wave64_arch_id(amdgcn_target_arch_id::HOST));
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

} // namespace ck_tile
