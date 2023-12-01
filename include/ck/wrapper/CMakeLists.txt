add_library(wrapper INTERFACE)
add_library(composable_kernel::wrapper ALIAS wrapper)

target_include_directories(wrapper INTERFACE
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/utility>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/tensor_description>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/ck/wrapper>
)
