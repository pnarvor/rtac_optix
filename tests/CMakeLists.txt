cmake_minimum_required(VERSION 3.10)
project(optix_helper_test VERSION 0.1)

list(APPEND test_names
    # base-test
    sources_test
    nvcrt-test
    context_test
    buffer_test
    raytype_test
    texture_base_test
    material_test
    model_test
    scene_test
)
list(APPEND test_deps
    optix_helpers
)
 
foreach(name ${test_names})
    add_executable(${PROJECT_NAME}_${name} src/${name}.cpp)
    target_link_libraries(${PROJECT_NAME}_${name} ${test_deps})
endforeach()

