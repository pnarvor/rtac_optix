
add_executable(mesh_test
    src/mesh_test.cpp
)
target_link_libraries(mesh_test PRIVATE
    rtac_optix
)
target_add_ptx(mesh_test
    CUDA_SOURCES src/mesh_test.cu
)
