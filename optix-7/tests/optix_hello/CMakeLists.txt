if(NOT TARGET rtac_cuda)
    find_package(rtac_cuda REQUIRED)
endif()

add_executable(optix_hello
    src/optix_hello.cpp
)
target_link_libraries(optix_hello PRIVATE
    rtac_optix
    rtac_cuda
)
target_add_ptx(optix_hello
    CUDA_SOURCES src/draw_solid_color.cu
)

