cmake_minimum_required(VERSION 3.10)
project(optix_helpers_display VERSION 0.1)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

if(NOT TARGET optix_helpers)
    find_package(optix_helpers REQUIRED)
endif()

list(APPEND optix_helpers_display_PUBLIC_HEADERS
    include/optix_helpers/display/GLBuffer.h
)
add_library(optix_helpers_display SHARED
    src/GLBuffer.cpp
)
set_target_properties(optix_helpers_display PROPERTIES 
    PUBLIC_HEADER "${optix_helpers_display_PUBLIC_HEADERS}"
    INSTALL_RPATH_USE_LINK_PATH True
)
target_include_directories(optix_helpers_display PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(optix_helpers_display PUBLIC
    optix_helpers
)

if(${BUILD_TESTS})
    add_subdirectory(tests)
endif()

rtac_install_target(optix_helpers_display)
