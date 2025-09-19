#include <gtest/gtest.h>
#include <string>
#include <sstream>

namespace ck_tile::test {

// Returns a string highlighting differences between actual and expected.
// Differences are enclosed in brackets with actual and expected parts separated by '|'.
std::string inlineDiff(const std::string& actual, const std::string& expected);

// A convenience alias for inlineDiff to improve readability in test assertions.
std::string formatInlineDiff(const std::string& actual, const std::string& expected);

} // namespace ck_tile::testing
