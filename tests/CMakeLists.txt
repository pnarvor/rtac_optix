cmake_minimum_required(VERSION 3.10)
project(optix_helper_test VERSION 0.1)

list(APPEND test_names
    # base-test
    # sources_test
    # nvcrt-test
    # context_test
    # buffer_test
    # raytype_test
    # material_test
    # model_test
    # ortho_test
    # pinhole_test
    # texture_base_test
    # texture_test
    # mirror_test
    scene_test
)
list(APPEND test_deps
    optix_helpers
)
if(WITH_DISPLAY)
    list(APPEND test_names
        animation_test
        render3d_test
    )
    list(APPEND test_deps
        optix_display
    )
endif()

foreach(name ${test_names})
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} ${test_deps})
endforeach()

# target_compile_definitions(nvcrt-test PRIVATE
#     NVRTC_COMPILE_OPTIONS="-arch compute_30 -use_fast_math -lineinfo -default-device -rdc true -D__x86_64"
# )
