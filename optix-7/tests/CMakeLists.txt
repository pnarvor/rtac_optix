add_executable(compile_test
    src/compile_test.cpp
    src/compile_test.cu
)
message(STATUS "Optix : ${OptiX}")
target_link_libraries(compile_test
    OptiX::OptiX
)
