list(APPEND gpu_list gfx908 gfx90a gfx940 gfx941 gfx942)
set(target 0)
foreach(gpu IN LISTS GPU_TARGETS)
    if(gpu IN_LIST gpu_list AND target EQUAL 0)
        add_custom_target(example_im2col_col2im)

        add_example_executable(example_image_to_column_f32 image_to_column_f32.cpp)
        add_example_dependencies(example_im2col_col2im example_image_to_column_f32)

        add_example_executable(example_column_to_image_f32 column_to_image_f32.cpp)
        add_example_dependencies(example_im2col_col2im example_column_to_image_f32)

        set(target 1)
    endif()
endforeach()
