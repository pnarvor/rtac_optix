#include <optix_helpers/Context.h>

namespace optix_helpers {

ContextObj::ContextObj(int entryPointCount) :
    context_(optix::Context::create())
{
    context_->setEntryPointCount(entryPointCount);
}

Program ContextObj::create_program(const Source& source, const Sources& additionalHeaders) const
{
    try {
        auto ptx = nvrtc_.compile(source, additionalHeaders);
        optix::Program program = context_->createProgramFromPTXString(ptx, source->name());
        return Program(new ProgramObj(source, additionalHeaders, program));
    }
    catch(const std::runtime_error& e) {
        std::ostringstream os;
        os << source <<  "\n" << e.what();
        throw std::runtime_error(os.str());
    }
}

Buffer ContextObj::create_buffer(RTbuffertype bufferType, RTformat format, 
                                 const std::string& name) const
{
    return Buffer(new BufferObj(context_->createBuffer(bufferType, format), name));
}

RayType ContextObj::create_raytype(const Source& rayDefinition) const
{
    unsigned int rayTypeIndex = context_->getRayTypeCount();
    context_->setRayTypeCount(rayTypeIndex + 1);
    return RayType(rayTypeIndex, rayDefinition);
}

Material ContextObj::create_material() const
{
    return Material(new MaterialObj(this->context()->createMaterial()));
}

Geometry ContextObj::create_geometry(const Program& intersection,
                                     const Program& boundingbox,
                                     size_t primitiveCount) const
{
    return Geometry(new GeometryObj(context_->createGeometry(),
                                    intersection, boundingbox,
                                    primitiveCount));
}

GeometryTriangles ContextObj::create_geometry_triangles(
    bool withTriangleIndexes, bool withNormals, bool withTextureCoordinates) const
{
    optix::Buffer vertexBuffer;
    optix::Buffer indexBuffer;
    optix::Buffer normalBuffer;
    optix::Buffer uvBuffer;

    vertexBuffer = context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3);
    if(withTriangleIndexes) {
        indexBuffer = context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3);
    }
    if(withNormals) {
        normalBuffer = context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3);
    }
    if(withTextureCoordinates) {
        uvBuffer = context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2);
    }
    return GeometryTriangles(new GeometryTrianglesObj(
        context_->createGeometryTriangles(),
        vertexBuffer, indexBuffer, normalBuffer, uvBuffer));
}

Model ContextObj::create_model() const
{
    return Model(new ModelObj(context_->createGeometryInstance()));
}

SceneItem ContextObj::create_scene_item(const Model& model, const char* acceleration) const
{
    return SceneItem(new SceneItemObj(context_->createGeometryGroup(),
                                      context_->createTransform(),
                                      context_->createAcceleration(acceleration),
                                      model));
}

//RayGenerator ContextObj::create_raygenerator(size_t width, size_t height, size_t depth) const
//{
//    return RayGenerator(width, height, depth, context_->createBuffer(RT_BUFFER_OUTPUT));
//}
//

optix::Handle<optix::VariableObj> ContextObj::operator[](const std::string& varname)
{
    return context_[varname];
}

ContextObj::operator optix::Context() const
{
    return context_;
}

optix::Context ContextObj::operator->()
{
    return context_;
}

optix::Context ContextObj::operator->() const
{
    return context_;
}

optix::Context ContextObj::context() const
{
    return context_;
}

Context::Context(int entryPointCount) :
    Handle<ContextObj>(new ContextObj(entryPointCount))
{}

} //namespace optix_helpers

