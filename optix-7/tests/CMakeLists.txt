add_executable(compile_test
    src/compile_test.cpp
    src/compile_test.cu
)
target_link_libraries(compile_test
    optix_helpers
)

list(APPEND test_names
    nvrtc_ptx.cpp
    optix_hello.cpp # from OptiX 7.1 SDK
)

list(APPEND test_deps
    optix_helpers
)

foreach(name ${test_names})

    # Generating a unique target name to avoid name colision with other rtac
    # packages.
    get_filename_component(executable_name ${name} NAME_WE)
    set(test_target_name ${PROJECT_NAME}_test_${executable_name})

    add_executable(${test_target_name} src/${name})
    target_link_libraries(${test_target_name} ${test_deps})
    set_target_properties(${test_target_name} PROPERTIES OUTPUT_NAME ${executable_name})

endforeach(name)



