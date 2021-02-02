add_executable(optix_hello
    src/optix_hello.cpp
)
target_link_libraries(optix_hello PRIVATE
    optix_helpers
)
target_add_ptx(optix_hello
    CUDA_SOURCES src/draw_solid_color.cu
)

