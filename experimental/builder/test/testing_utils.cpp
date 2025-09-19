#include <string>
#include <sstream>
#include <vector>
#include <algorithm>

namespace ck_tile::test {

// Copilot-generated diff function, seems to work great.
// TODO: Study this, maybe add unit tests?
//
// This implementation is based on the Wagner-Fischer algorithm to find the
// optimal edit script.
std::string inlineDiff(const std::string& actual, const std::string& expected)
{
    const size_t n = expected.length();
    const size_t m = actual.length();

    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1));

    for(size_t i = 0; i <= n; ++i)
    {
        dp[i][0] = i;
    }
    for(size_t j = 0; j <= m; ++j)
    {
        dp[0][j] = j;
    }

    for(size_t i = 1; i <= n; ++i)
    {
        for(size_t j = 1; j <= m; ++j)
        {
            int cost = (expected[i - 1] == actual[j - 1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i - 1][j] + 1,        // Deletion
                                 dp[i][j - 1] + 1,        // Insertion
                                 dp[i - 1][j - 1] + cost}); // Substitution/Match
        }
    }

    std::ostringstream diff;
    std::string expected_diff, actual_diff;
    bool in_diff = false;

    size_t i = n, j = m;
    while(i > 0 || j > 0)
    {
        int cost = (i > 0 && j > 0 && expected[i - 1] == actual[j - 1]) ? 0 : 1;

        if(i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + cost)
        {
            if(cost == 0)
            {
                if(in_diff)
                {
                    std::reverse(expected_diff.begin(), expected_diff.end());
                    std::reverse(actual_diff.begin(), actual_diff.end());
                    diff << "]" << expected_diff << "|" << actual_diff << "[";
                    expected_diff.clear();
                    actual_diff.clear();
                    in_diff = false;
                }
                diff << expected[i - 1];
            }
            else
            {
                in_diff = true;
                expected_diff += expected[i - 1];
                actual_diff += actual[j - 1];
            }
            --i;
            --j;
        }
        else if(j > 0 && dp[i][j] == dp[i][j - 1] + 1)
        {
            in_diff = true;
            actual_diff += actual[j - 1];
            --j;
        }
        else if(i > 0 && dp[i][j] == dp[i - 1][j] + 1)
        {
            in_diff = true;
            expected_diff += expected[i - 1];
            --i;
        }
    }

    if(in_diff)
    {
        std::reverse(expected_diff.begin(), expected_diff.end());
        std::reverse(actual_diff.begin(), actual_diff.end());
        diff << "]" << expected_diff << "|" << actual_diff << "[]";
    }

    std::string result = diff.str();
    std::reverse(result.begin(), result.end());
    return result;
}

std::string formatInlineDiff(const std::string& actual, const std::string& expected)
{
    return std::string("Inline diff:  \"") + inlineDiff(actual, expected) + "\"";
}

} // namespace ck_tile::test
