
set(target_name ${PROJECT_NAME}_materials_test)

add_executable(${target_name}
    src/materials_test.cpp
)
target_link_libraries(${target_name} PRIVATE
    rtac_optix
)
target_add_ptx(${target_name}
    CUDA_SOURCES src/materials_test.cu
)
