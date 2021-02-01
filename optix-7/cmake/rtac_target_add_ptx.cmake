function(rtac_target_add_ptx TARGET_NAME)
    
    message(STATUS "argument : ${ARGN}")
	cmake_parse_arguments(ARGUMENTS "TAG_FOR_INSTALL" "OUTPUT_NAME;INSTALL_DESTINATION" "CUDA_SOURCES" ${ARGN} )

    # output filename
    if("${ARGUMENTS_OUTPUT_NAME}" STREQUAL "")
        set(output_name "${TARGET_NAME}_ptx_files.h")
    else()
        set(output_name "${ARGUMENTS_OUTPUT_NAME}")
    endif()

    if("${ARGUMENTS_INSTALL_DESTINATION}" STREQUAL "")
        set(ARGUMENTS_INSTALL_DESTINATION "${TARGET_NAME}")
    endif()

    enable_language(CUDA)
    if(NOT TARGET OptiX::OptiX)
        find_package(OptiX REQUIRED)
    endif()
    
    # Creating an OBJECT target to generate the ptx files.
    set(ptx_target ${TARGET_NAME}_PTX_GEN)
    add_library(${ptx_target} OBJECT ${ARGUMENTS_CUDA_SOURCES})
    target_link_libraries(${ptx_target}
        OptiX::OptiX
    )
    set_target_properties(${ptx_target} PROPERTIES
        CUDA_PTX_COMPILATION ON
    )
    # The ptx files must be generated before TARGET_NAME
    add_dependencies(${TARGET_NAME} ${ptx_target})

    # After the generation of the PTX files but before the compilation of
    # TARGET_NAME, PTX files will be aggregated into a single header file. The
    # PTX code will be integrated directly into the compiled binaries, so
    # there is no runtime-dependency on installed headers.
    set(ptx_target_output_dir "${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${ptx_target}.dir")

    # Generating the list of path to ptx files which will be generated 
    foreach(source ${ARGUMENTS_CUDA_SOURCES})
        
        get_filename_component(source_dir ${source} DIRECTORY)
        get_filename_component(name_wle   ${source} NAME_WLE)
        
        # Generating expected output ptx file name
        if(IS_ABSOLUTE ${source})
            set(ptx_output_file "${ptx_target_output_dir}${source_dir}/${name_wle}.ptx")
        else()
            set(ptx_output_file "${ptx_target_output_dir}/${source_dir}/${name_wle}.ptx")
        endif()
        list(APPEND ptx_files ${ptx_output_file})

    endforeach()
   
    # Command responsible to include all compilated PTX file into a C++ header
    # file. PTX strings can be retrieved from a filename keyed dictionary.
    add_custom_command(OUTPUT ${ptx_target_output_dir}/${output_name}
                       COMMAND ${CMAKE_COMMAND}
                       -DSOURCE_FILES="${ARGUMENTS_CUDA_SOURCES}"
                       -DPTX_FILES="${ptx_files}"
                       -DTARGET_NAME=${TARGET_NAME}
                       -DOUTPUT_FILE=${ptx_target_output_dir}/${output_name}
                       -P ${CMAKE_SOURCE_DIR}/cmake/generate_ptx_header.cmake
                       COMMENT "Generating PTX header file ${output_name}")
    set(ptx_header_target ${TARGET_NAME}_PTX_GEN_HEADER)
    add_custom_target(${ptx_header_target}
        DEPENDS ${ptx_target_output_dir}/${output_name}
    )
    add_dependencies(${ptx_header_target} ${ptx_target})

    # Adding output directory to include directories for compilation
    target_include_directories(${TARGET_NAME} PUBLIC
        $<BUILD_INTERFACE:${ptx_target_output_dir}>
    )
    # Adding a dependency to ensure header generation before compilation of
    # TARGET_NAME
    add_dependencies(${TARGET_NAME} ${ptx_header_target})

    if(ARGUMENTS_TAG_FOR_INSTALL)
        install(FILES ${ptx_target_output_dir}/${output_name}
                DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${ARGUMENTS_INSTALL_DESTINATION})
    endif()

endfunction()






