#include <rtac_optix/GeometryAccelStruct.h>

namespace rtac { namespace optix {

GeometryAccelStruct::BuildInput GeometryAccelStruct::default_build_input()
{
    return AccelerationStruct::default_build_input();
}

GeometryAccelStruct::BuildOptions GeometryAccelStruct::default_build_options()
{
    return AccelerationStruct::default_build_options();
}

std::vector<unsigned int> GeometryAccelStruct::default_hit_flags()
{
    return std::vector<unsigned int>({OPTIX_GEOMETRY_FLAG_NONE});
}

GeometryAccelStruct::GeometryAccelStruct(const Context::ConstPtr& context,
                                         const OptixBuildInput& buildInput,
                                         const OptixAccelBuildOptions& options) :
    AccelerationStruct(context, buildInput, options)
{
    this->material_hit_setup(default_hit_flags());
}

void GeometryAccelStruct::update_hit_setup() const
{
    switch(this->buildInput_.type)
    {
        case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
            if(materialHitFlags_.size() > 0) {
                this->buildInput_.triangleArray.flags         = materialHitFlags_.data();
                this->buildInput_.triangleArray.numSbtRecords = materialHitFlags_.size();
            }
            else {
                this->buildInput_.triangleArray.flags         = nullptr;
                this->buildInput_.triangleArray.numSbtRecords = 0;
            }
            if(materialIndexes_) {
                if(materialIndexes_->size() != this->primitive_count()) {
                    std::ostringstream oss;
                    oss << "GeometryAccelStruct::material_hit_setup : "
                        << "Invalid number of material indexes, must be the same than "
                        << "the number of primitives (got " << materialIndexes_->size()
                        << ", expected " << this->primitive_count() << ")";
                    throw std::runtime_error(oss.str());
                }
                this->buildInput_.triangleArray.sbtIndexOffsetBuffer =
                    (CUdeviceptr)materialIndexes_->data();
                this->buildInput_.triangleArray.sbtIndexOffsetSizeInBytes   = 1;
                this->buildInput_.triangleArray.sbtIndexOffsetStrideInBytes = 1;
            }
            break;
        case OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES:
            if(materialHitFlags_.size() > 0) {
                this->buildInput_.customPrimitiveArray.flags         = materialHitFlags_.data();
                this->buildInput_.customPrimitiveArray.numSbtRecords = materialHitFlags_.size();
            }
            else {
                this->buildInput_.customPrimitiveArray.flags         = nullptr;
                this->buildInput_.customPrimitiveArray.numSbtRecords = 0;
            }
            break;
        case OPTIX_BUILD_INPUT_TYPE_CURVES:
            throw std::logic_error("Curves not implemented yet.");
            break;
        case OPTIX_BUILD_INPUT_TYPE_INSTANCES:
        case OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS:
            throw std::runtime_error(
                "Invalid buildInput type (Instances) for GeometryAccelStruct");
            break;
        default:
            throw std::logic_error("Fatal error : Unknown build type. Check OptiX version");
            break;
    };
}

void GeometryAccelStruct::do_build() const
{
    this->update_hit_setup();
    AccelerationStruct::do_build();
}

void GeometryAccelStruct::material_hit_setup(
    const std::vector<unsigned int>& hitFlags,
    const Handle<MaterialIndexBuffer>& materialIndexes)
{
    if(hitFlags.size() == 0) {
        this->bump_version();
        materialHitFlags_.resize(0);
    }
    else if(hitFlags.size() == 1) {
        if(materialIndexes) {
            std::cerr << "Warning, GeomAccelStruct : You provided a materialIndexes "
                      << "buffer but requested only one material type. "
                      << "The indexes will be ignored";
        }
        this->bump_version();
        materialHitFlags_ = hitFlags;
    }
    else if(hitFlags.size() > 1) {
        if(!materialIndexes) {
            std::ostringstream oss;
            oss << "GeometryAccelStruct::material_hit_setup : "
                << "several materials hit flags where provided (implying your"
                << "geometry has several materials), but no material index "
                << "buffer was provided (buffer size must be number of "
                << "primitives in the geometry, buffers value must be a "
                << "material index).";
            throw std::runtime_error(oss.str());
        }
        if(materialIndexes->size() != this->primitive_count()) {
            std::ostringstream oss;
            oss << "GeometryAccelStruct::material_hit_setup : "
                << "Invalid number of material indexes, must be the same than "
                << "the number of primitives (got " << materialIndexes->size()
                << ", expected " << this->primitive_count() << ")";
            throw std::runtime_error(oss.str());
        }
        this->bump_version();
        materialHitFlags_ = hitFlags;
        materialIndexes_  = materialIndexes;
    }
}

void GeometryAccelStruct::material_hit_setup(
    const std::vector<unsigned int>& hitFlags,
    const std::vector<uint8_t>& materialIndexes)
{
    this->material_hit_setup(hitFlags,
        Handle<MaterialIndexBuffer>(new MaterialIndexBuffer(materialIndexes)));
}

void GeometryAccelStruct::clear_hit_setup()
{
    this->bump_version();
    materialHitFlags_.resize(0);
}

unsigned int GeometryAccelStruct::sbt_width() const
{
    return materialHitFlags_.size();
}

}; //namespace optix
}; //namespace rtac
