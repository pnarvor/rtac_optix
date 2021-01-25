cmake_minimum_required(VERSION 3.10)
project(rtac_optix_7 VERSION 0.1 LANGUAGES CUDA CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OptiX REQUIRED)

if(NOT TARGET OptiX)
    message(STATUS "OptiX is not a target")
else()
    message(STATUS "OptiX is a target")
endif()
if(NOT TARGET OptiX::OptiX)
    message(STATUS "OptiX::OptiX is not a target")
else()
    message(STATUS "OptiX::OptiX is a target")
endif()

add_subdirectory(tests)


