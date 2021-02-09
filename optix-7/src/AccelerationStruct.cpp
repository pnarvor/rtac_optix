#include <rtac_optix/AccelerationStruct.h>

namespace rtac { namespace optix {

AccelerationStruct::AccelerationStruct(const Context::ConstPtr& context) :
    context_(context),
    handle_(0),
    buffer_(0)
{}

AccelerationStruct::Ptr AccelerationStruct::Create(const Context::ConstPtr& context)
{
    return Ptr(new AccelerationStruct(context));
}

void AccelerationStruct::build(const OptixBuildInput& buildInput,
                              const OptixAccelBuildOptions& buildOptions,
                              Buffer& tempBuffer, CUstream cudaStream)
{
    // Computing memory usage needed(both for output and temporary usage for
    // the build itself) and resizing buffers accordingly;
    OptixAccelBufferSizes bufferSizes; // should I keep that in attributes ?
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        *context_, &buildOptions, &buildInput, 1, &bufferSizes) );

    if(!(buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION)) {
        // if compaction is not requested, building and exiting right away
        tempBuffer.resize(bufferSizes.tempSizeInBytes);
        buffer_.resize(bufferSizes.outputSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(*context_, cudaStream,
            &buildOptions, &buildInput, 1,
            reinterpret_cast<CUdeviceptr>(tempBuffer.data()), tempBuffer.size(),
            reinterpret_cast<CUdeviceptr>(buffer_.data()), buffer_.size(),
            &handle_, nullptr, 0));
    }
    else {
        // Compaction is requested, a second temporary space is needed.
        // using tempBuffer as a single temp memory space.
        // /!\ CATCH : memory space pointers must be 128bit aligned.
        // The begining of the second temporary space might not start directly
        // at the end of the first.
        // Also, output compacted size is returned by the build operation in
        // device memory, so an extra memory space must be reserved (size 64bits).

        // This function will compute the successive offsets at which the
        // output buffer and the compacted size may start.
        auto offsets = compute_aligned_offsets<2,size_t>(
            {bufferSizes.tempSizeInBytes, bufferSizes.outputSizeInBytes},
            128 / 8 );//128bits aligned

        // Compacted size will be returned as a uint64_t
        // (Total needed size is last offset + last size
        tempBuffer.resize(offsets.back() + sizeof(uint64_t));
        
        // Building the request for the compacted size which will be send to
        // optixAccelBuild.
        auto propertyRequest   = zero<OptixAccelEmitDesc>();
        propertyRequest.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        propertyRequest.result = reinterpret_cast<CUdeviceptr>(
            tempBuffer.data() + offsets.back());
            
        OPTIX_CHECK(optixAccelBuild(*context_, cudaStream,
            &buildOptions, &buildInput, 1,
            reinterpret_cast<CUdeviceptr>(tempBuffer.data()), offsets[0],
            reinterpret_cast<CUdeviceptr>(tempBuffer.data() + offsets[0]),
            offsets[1] - offsets[0],
            &handle_, &propertyRequest, 1));

        // Retrieving compacted size
        uint64_t compactedSize;
        cudaMemcpy(&compactedSize, reinterpret_cast<const void*>(propertyRequest.result),
                   sizeof(compactedSize), cudaMemcpyDeviceToHost);
        
        // This check prevent the compaction if there is no gain to compact the
        // data.  However, in this implementation it will require to move the
        // data after completion if no compaction is possible and this feature
        // is not implemented yet. So the compaction executed in any case.
        //if(compactedSize < bufferSizes.outputSizeInBytes)
        {
            buffer_.resize(compactedSize);
            OPTIX_CHECK(optixAccelCompact(*context_, cudaStream, handle_,
                reinterpret_cast<CUdeviceptr>(buffer_.data()), buffer_.size(),
                &handle_));
        }
    }
}

void AccelerationStruct::build(const OptixBuildInput& buildInput,
                              const OptixAccelBuildOptions& buildOptions,
                              CUstream cudaStream)
{
    Buffer tempBuffer;
    this->build(buildInput, buildOptions, tempBuffer, cudaStream);
    // the build operation is asynchronous. A sync barrier is required here
    // because tempBuffer will go out of scope and the associated device memory
    // will be released. This must not happen before the end of the build.
    cudaStreamSynchronize(cudaStream);
}

AccelerationStruct::operator OptixTraversableHandle() const
{
    return handle_;
}

CUdeviceptr AccelerationStruct::data()
{
    return reinterpret_cast<CUdeviceptr>(buffer_.data());
}

}; //namespace optix
}; //namespace rtac
