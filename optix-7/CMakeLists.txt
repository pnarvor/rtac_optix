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

list(APPEND rtac_optix_PUBLIC_HEADERS
    include/rtac_optix/Handle.h
    include/rtac_optix/Source.h
    include/rtac_optix/Nvrtc.h
    include/rtac_optix/Pipeline.h
)
list(APPEND rtac_optix_SOURCES
    src/Source.cpp
    src/Nvrtc.cpp
    src/Pipeline.cpp
)

add_library(rtac_optix SHARED ${rtac_optix_SOURCES})
set_target_properties(rtac_optix PROPERTIES 
    RTAC_PUBLIC_HEADERS "${rtac_optix_PUBLIC_HEADERS}"
    INSTALL_RPATH_USE_LINK_PATH True
)
target_include_directories(rtac_optix
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)
target_link_libraries(rtac_optix PUBLIC
    OptiX::OptiX
    rtac_base
    rtac_cuda
)

rtac_install_target(rtac_optix
    ADDITIONAL_CONFIG_FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindOptiX.cmake
)

add_subdirectory(tests)


