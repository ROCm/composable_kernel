# Tests for Tutorial 07: Tile Sweeping with Y-Dimension Repetition

# Test A distribution with Y-repetition
add_executable(test_a_distribution_y_repetition test_a_distribution_y_repetition.cpp)
target_include_directories(test_a_distribution_y_repetition PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

# Test B distribution with Y-repetition
add_executable(test_b_distribution_y_repetition test_b_distribution_y_repetition.cpp)
target_include_directories(test_b_distribution_y_repetition PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

# Test B Y-slicing with get_y_sliced_thread_data
add_executable(test_b_y_slicing test_b_y_slicing.cpp)
target_include_directories(test_b_y_slicing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

# Test A Y-slicing with get_y_sliced_thread_data
add_executable(test_a_y_slicing test_a_y_slicing.cpp)
target_include_directories(test_a_y_slicing PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
)

message(STATUS "Added Tutorial 07 tests: A and B distributions with Y-repetition, A and B Y-slicing")
