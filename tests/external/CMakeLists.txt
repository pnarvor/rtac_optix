cmake_minimum_required(VERSION 3.10)
project(optix_external)

find_package(optix_helpers REQUIRED)

add_executable(optix_external
    src/base-test.cpp
)
target_link_libraries(optix_external
    PRIVATE optix_helpers
)
