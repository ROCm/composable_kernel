set(SPACE_FILLING_CURVE_SOURCE
    space_filling_curve.cpp
)

get_filename_component(SPACE_FILLING_CURVE ${SPACE_FILLING_CURVE_SOURCE} NAME_WE)
add_test_executable(test_${SPACE_FILLING_CURVE} ${SPACE_FILLING_CURVE_SOURCE})
