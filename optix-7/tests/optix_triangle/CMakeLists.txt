

add_executable(optix_triangle
    src/optix_triangle.cpp
)
target_link_libraries(optix_triangle PRIVATE
    rtac_optix
)
target_add_ptx(optix_triangle
    CUDA_SOURCES src/optix_triangle.cu
)


