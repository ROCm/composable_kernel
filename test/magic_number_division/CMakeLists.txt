set(MAGIC_NUMBER_DIVISION_SOURCE
    magic_number_division.cpp
)

get_filename_component(MAGIC_NUMBER_DIVISION ${MAGIC_NUMBER_DIVISION_SOURCE} NAME_WE)
add_test_executable(test_${MAGIC_NUMBER_DIVISION} ${MAGIC_NUMBER_DIVISION_SOURCE})
target_link_libraries(test_${MAGIC_NUMBER_DIVISION} PRIVATE host_tensor)
