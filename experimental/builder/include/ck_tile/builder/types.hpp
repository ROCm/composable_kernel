#pragma once

namespace ck_tile::builder {

enum class DataType
{
    FP64,
    FP32,
    FP16,
    BF16,
    S16,
    S8,
    S4,
};

// Helper to provide a readable error for unsupported data types.
// The compiler will print the name of this struct in the error message.
template <DataType T>
struct unsupported_data_type
{
};

} // namespace ck_tile::builder
