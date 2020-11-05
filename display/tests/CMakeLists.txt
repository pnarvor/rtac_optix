cmake_minimum_required(VERSION 3.10)
project(optix_helper_display_test VERSION 0.1)

find_package(rtac_display REQUIRED)

list(APPEND test_names
    animation_test
    render3d_test
)
list(APPEND test_deps
    optix_helpers_display
    rtac_display
)
 
foreach(name ${test_names})
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} ${test_deps})
endforeach()

