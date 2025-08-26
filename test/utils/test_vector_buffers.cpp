// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gtest/gtest.h"

#include "ck/utility/common_header.hpp"
#include "ck/utility/sequence.hpp"
#include "ck/utility/sequence_helper.hpp"
#include "ck/utility/number.hpp"
#include "ck/utility/amd_address_space.hpp"
#include "ck/tensor/static_tensor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"

using namespace ck;

TEST(StaticBufferTupleOfVector, TestConstruction)
{
  constexpr index_t num_vector = 8;
  constexpr index_t vector_size = 2;
  StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                        ck::bhalf_t,
                                        num_vector,
                                        vector_size,
                                        true> buffer;
  EXPECT_EQ(buffer.s_per_v, vector_size);
  EXPECT_EQ(buffer.num_of_v_, num_vector);
  EXPECT_EQ(buffer.s_per_buf, num_vector * vector_size);
}

TEST(StaticBufferTupleOfVector, TestAccess)
{
  constexpr auto I2 = Number<2>{};

  constexpr index_t num_vector = 2;
  constexpr index_t vector_size = 2;
  StaticBufferTupleOfVector<AddressSpaceEnum::Vgpr,
                                        ck::bhalf_t,
                                        num_vector,
                                        vector_size,
                                        true> buffer;
  
  const ck::bhalf2_t val1{1, 2};
  const ck::bhalf2_t val2{3, 4};

  static_for<0, num_vector, 1>{}([&](auto i_pair) 
  {
      const auto val = (i_pair == 0) ? val1 : val2;
      constexpr auto offset = I2 * i_pair;
      buffer.template SetAsType<ck::bhalf2_t>(Number<offset>{}, val);
  });

  // Test vectorized read and verify values
  static_for<0, num_vector, 1>{}([&](auto i_pair) 
  {
      constexpr auto offset = I2 * i_pair;
      const auto read_val = buffer.template GetAsType<ck::bhalf2_t>(Number<offset>{});
      if constexpr(i_pair == 0)
      {
          EXPECT_EQ(read_val[0], val1[0]);
          EXPECT_EQ(read_val[1], val1[1]);
      }
      else
      {
          EXPECT_EQ(read_val[0], val2[0]);
          EXPECT_EQ(read_val[1], val2[1]);
      }
  });
}
