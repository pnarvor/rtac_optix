#include <optix_helpers/Context.h>

namespace optix_helpers {

ContextObj::ContextObj() :
    context_(optix::Context::create())
{
}


optix::Context ContextObj::context() const
{
    return context_;
}

unsigned int ContextObj::num_raytypes() const
{
    return this->context()->getRayTypeCount();
}

Program ContextObj::create_program(const Source& source, const Sources& additionalHeaders) const
{
    try {
        auto ptx = nvrtc_.compile(source, additionalHeaders);
        optix::Program program = context_->createProgramFromPTXString(ptx, source->name());
        return Program(source, additionalHeaders, program);
    }
    catch(const std::runtime_error& e) {
        std::ostringstream os;
        os << source <<  "\n" << e.what();
        throw std::runtime_error(os.str());
    }
}

RayGenerationProgram ContextObj::create_raygen_program(const std::string renderBufferName,
                                                       const Source& source,
                                                       const Sources& additionalHeaders) const
{
    return RayGenerationProgram(this->create_program(source, additionalHeaders),
                                renderBufferName);
}

RayType ContextObj::create_raytype(const Source& rayDefinition) const
{
    unsigned int rayTypeIndex = this->num_raytypes();
    this->context()->setRayTypeCount(rayTypeIndex + 1);
    return RayType(rayTypeIndex, rayDefinition);
}

Material ContextObj::create_material() const
{
    return Material(this->context()->createMaterial());
}

Geometry ContextObj::create_geometry(const Program& intersection,
                                     const Program& boundingbox,
                                     size_t primitiveCount) const
{
    return Geometry(context_->createGeometry(), intersection, boundingbox, primitiveCount);
}

GeometryTriangles ContextObj::create_geometry_triangles() const
{
    return GeometryTriangles(context_->createGeometryTriangles());
}

Model ContextObj::create_model() const
{
    return Model(context_->createGeometryInstance());
}

RayGenerator ContextObj::create_raygenerator(size_t width, size_t height, size_t depth) const
{
    return RayGenerator(width, height, depth, context_->createBuffer(RT_BUFFER_OUTPUT));
}

Context::Context() :
    Handle<ContextObj>(new ContextObj)
{}

} //namespace optix_helpers

