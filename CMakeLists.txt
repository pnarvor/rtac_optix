cmake_minimum_required(VERSION 3.10)
project(optix_interface VERSION 0.1)

find_path(OPTIX_PREFIX "include/optix.h"
    HINTS ${OPTIX_PATH}
)
find_path(OPTIX_INCLUDE_DIR "optix.h"
    HINTS ${OPTIX_PREFIX}/include
)
find_library(optix_LIBRARY "optix"
    HINTS ${OPTIX_PREFIX}
    PATH_SUFFIXES lib lib64
)
find_library(optixu_LIBRARY "optixu"
    HINTS ${OPTIX_PREFIX}
    PATH_SUFFIXES lib lib64
)
find_library(optix_prime_LIBRARY "optix_prime"
    HINTS ${OPTIX_PREFIX}
    PATH_SUFFIXES lib lib64
)

if(${OPTIX_PREFIX} STREQUAL "OPTIX_PREFIX-NOTFOUND")
    message(FATAL_ERROR "Could not find OptiX SDK inside \"${OPTIX_PATH}\" folder. Please check the -DOPTIX_PATH option.")
endif()
message(STATUS "Found OptiX         : ${OPTIX_PREFIX}")
message(STATUS "OptiX include dirs  : ${OPTIX_INCLUDE_DIR}")
message(STATUS "optix library       : ${optix_LIBRARY}")
message(STATUS "optixu library      : ${optixu_LIBRARY}")
message(STATUS "optix_prime library : ${optix_prime_LIBRARY}")

add_library(optix INTERFACE)
set_target_properties(optix PROPERTIES 
    INTERFACE_INCLUDE_DIRECTORIES ${OPTIX_INCLUDE_DIR}
    INTERFACE_LINK_LIBRARIES "${optix_LIBRARY};${optixu_LIBRARY};${optix_prime_LIBRARY}"
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/installation.cmake)
install_target(optix)



