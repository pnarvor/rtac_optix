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

    # Generating a unique target name to avoid name colision with other rtac
    # packages.
    get_filename_component(executable_name ${name} NAME_WE)
    set(test_target_name ${PROJECT_NAME}_test_${executable_name})

    add_executable(${test_target_name} src/${name})
    target_link_libraries(${test_target_name} ${test_deps})
    set_target_properties(${test_target_name} PROPERTIES OUTPUT_NAME ${executable_name})

endforeach()

