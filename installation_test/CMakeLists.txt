cmake_minimum_required(VERSION 3.10)
project(optix_helpers_installation_check VERSION 0.1)


find_package(optix_helpers_display REQUIRED)

add_executable(optix_helpers_installation_check
    src/main.cpp
)
target_link_libraries(optix_helpers_installation_check
    PUBLIC
        optix_helpers_display
)
