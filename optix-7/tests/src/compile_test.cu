#include <optix.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__device__
void device_copy(const float& v1, float& v2)
{
    v2 = v1;
}

__global__
void global_copy(const float* input, float* output)
{
    device_copy(input[threadIdx.x], output[threadIdx.x]);
}

void copy(const std::vector<float>& input, std::vector<float>& output)
{
    thrust::device_vector<float> in(input);
    thrust::device_vector<float> out(in.size());

    global_copy<<<1,in.size()>>>(thrust::raw_pointer_cast(in.data()),
                                 thrust::raw_pointer_cast(out.data()));
    cudaDeviceSynchronize();

    output.resize(out.size());

    cudaMemcpy(output.data(), thrust::raw_pointer_cast(out.data()),
               sizeof(float)*out.size(), cudaMemcpyDeviceToHost);
}




