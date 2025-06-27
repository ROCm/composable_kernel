#include "smoothquant.inc"

int main()
{
    std::vector<std::vector<std::string>> test_cases = create_test_cases("fp16");

    return !run_test_cases<ck_tile::half_t>(test_cases);
}
