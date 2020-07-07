cmake_minimum_required(VERSION 3.10)
project(optix_helper_test VERSION 0.1)

get_target_property(definitions optix_helpers INTERFACE_COMPILE_DEFINITIONS)
message(STATUS "definitions : ${definitions}")

list(APPEND test_names
    nvcrt-test
)
list(APPEND test_deps
    optix_helpers
)

foreach(name ${test_names})
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} ${test_deps})
endforeach()
