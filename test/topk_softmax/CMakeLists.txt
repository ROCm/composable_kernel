add_test_executable(test_topk_softmax topk_softmax.cpp ${CMAKE_CURRENT_SOURCE_DIR}/../../example/ck_tile/05_moe/topk_softmax_api.cpp)
target_include_directories(test_topk_softmax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../example/ck_tile/05_moe)
target_compile_options(test_topk_softmax PRIVATE -v --save-temps -Wno-gnu-line-marker)
