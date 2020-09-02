cmake_minimum_required(VERSION 3.10)
project(optix_helper_test VERSION 0.1)

find_package(rtac_tools REQUIRED)

list(APPEND test_names
    nvcrt-test
    # sources_test
    # context_test
    # base-test
    # raytype_test
    # material_test
    # model_test
    # render_test
    item_test
)
list(APPEND test_deps
    optix_helpers
    rtac_tools
)

foreach(name ${test_names})
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} ${test_deps})
endforeach()

target_compile_definitions(nvcrt-test PRIVATE
    NVRTC_COMPILE_OPTIONS="-arch compute_30 -use_fast_math -lineinfo -default-device -rdc true -D__x86_64"
)
