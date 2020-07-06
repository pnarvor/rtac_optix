cmake_minimum_required(VERSION 3.10)
project(optix_package_test VERSION 0.1)

find_package(optix REQUIRED)

add_executable(optix_package_test
    src/main.cpp
)
target_link_libraries(optix_package_test
    optix
)
