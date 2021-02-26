

set(target_name ${PROJECT_NAME}_autosbt_test)

add_executable(${target_name}
    src/autosbt_test.cpp
)
target_link_libraries(${target_name} PRIVATE rtac_optix)
target_add_ptx(${target_name}
    CUDA_SOURCES src/autosbt_test.cu
)

