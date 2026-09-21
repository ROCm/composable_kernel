add_custom_target(test_synchronization)

add_gtest_executable(test_monitor_mwait monitor_mwait.cpp)
if(result EQUAL 0)
    target_link_libraries(test_monitor_mwait PRIVATE utility)
    add_dependencies(test_synchronization test_monitor_mwait)
endif()


add_gtest_executable(test_async_lds_load_store async_lds_load_store.cpp)
if(result EQUAL 0)
    target_link_libraries(test_async_lds_load_store PRIVATE utility)
    add_dependencies(test_synchronization test_async_lds_load_store)
endif()


