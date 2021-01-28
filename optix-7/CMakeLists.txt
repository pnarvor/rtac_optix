cmake_minimum_required(VERSION 3.10)
project(rtac_optix_7 VERSION 0.1 LANGUAGES CUDA CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(OptiX REQUIRED)

if(NOT TARGET rtac_base)
    find_package(rtac_base REQUIRED)
endif()
if(NOT TARGET rtac_cuda)
    find_package(rtac_cuda REQUIRED)
endif()

list(APPEND optix_helpers_PUBLIC_HEADERS
    include/optix_helpers/Handle.h
    include/optix_helpers/Source.h
    include/optix_helpers/Nvrtc.h
)
list(APPEND optix_helpers_SOURCES
    src/Source.cpp
    src/Nvrtc.cpp
)

add_library(optix_helpers SHARED ${optix_helpers_SOURCES})
set_target_properties(optix_helpers PROPERTIES 
    RTAC_PUBLIC_HEADERS "${optix_helpers_PUBLIC_HEADERS}"
    INSTALL_RPATH_USE_LINK_PATH True
)
target_include_directories(optix_helpers
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(optix_helpers PUBLIC
    OptiX::OptiX
    rtac_base
    rtac_cuda
)

rtac_install_target(optix_helpers
    ADDITIONAL_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindOptiX.cmake
)

add_subdirectory(tests)


