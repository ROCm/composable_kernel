# Test programs for Tutorial 06 tile distributions

# Test A distribution with replication
add_executable(test_a_dist_replication test_a_distribution_with_replication.cpp)
target_include_directories(test_a_dist_replication PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)
# target_compile_options(test_a_dist_replication PRIVATE -Wall -O0 -g)

# Test B distribution with replication
add_executable(test_b_dist_replication test_b_distribution_with_replication.cpp)
target_include_directories(test_b_dist_replication PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)
# target_compile_options(test_b_dist_replication PRIVATE -Wall -O0 -g)

# Test A distribution using embed API
add_executable(test_a_embed test_a_embed_distribution.cpp)
target_include_directories(test_a_embed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)
# target_compile_options(test_a_embed PRIVATE -Wall -O0 -g)

# Test B distribution using embed API
add_executable(test_b_embed test_b_embed_distribution.cpp)
target_include_directories(test_b_embed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)
# target_compile_options(test_b_embed PRIVATE -Wall -O0 -g)

message(STATUS "Added Tutorial 06 distribution tests (with and without embed API)")
