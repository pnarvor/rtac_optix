

add_executable(pinhole_test
    src/pinhole_test.cpp
)
target_link_libraries(pinhole_test PRIVATE
    rtac_optix
)
target_add_ptx(pinhole_test
    CUDA_SOURCES src/pinhole_test.cu
)

