cmake_minimum_required(VERSION 3.10)
project(optix_helper_display_test VERSION 0.1)

if(NOT TARGET rtac_display)
    find_package(rtac_display REQUIRED)
endif()

list(APPEND test_names
    animation_test
    render3d_test
)
list(APPEND test_deps
    optix_helpers_display
    rtac_display
)
 
foreach(name ${test_names})
    add_executable(${PROJECT_NAME}_${name} src/${name}.cpp)
    target_link_libraries(${PROJECT_NAME}_${name} ${test_deps})
endforeach()

