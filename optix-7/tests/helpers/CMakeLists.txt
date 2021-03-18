

set(target_name ${PROJECT_NAME}_helpers)
add_executable(${target_name}
    src/helpers.cpp
)
target_link_libraries(${target_name} PRIVATE
    rtac_optix
)
target_add_ptx(${target_name}
    CUDA_SOURCES src/helpers.cu
)

