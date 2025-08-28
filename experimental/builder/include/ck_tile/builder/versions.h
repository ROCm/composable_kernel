#pragma once

#include <concepts>
#include <string_view>

namespace ck_tile::builder {

static constexpr char V0_0_0[] = "0.0.0";
static constexpr char V0_1_0[] = "0.1.0";

template <const char* V>
concept SupportedVersion = (std::string_view{V} == std::string_view{V0_0_0}) ||
                           (std::string_view{V} == std::string_view{V0_1_0});

} // namespace ck_tile::builder
