#include <iostream>
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/conv_builder.hpp"



namespace
{
    std::string to_string(const ck_tile::builder::DataType& dt)
    {
        switch(dt)
        {
            case ck_tile::builder::DataType::FP16: return "FP16";
            case ck_tile::builder::DataType::BF16: return "BF16";
            case ck_tile::builder::DataType::FP32: return "FP32";
            case ck_tile::builder::DataType::FP64: return "FP64";
            case ck_tile::builder::DataType::S16: return "S16";
            case ck_tile::builder::DataType::S4: return "S4";
            case ck_tile::builder::DataType::S8: return "S8";
            default: return "Unknown";
        }
    }
} // namespace

int main() {

  using namespace ck_tile::builder;

  const auto dt = DataType::BF16;

  std::cout << "Hello, builder!" << std::endl;
  std::cout << "DataType: " << to_string(dt) << std::endl;
  return 0;
}
