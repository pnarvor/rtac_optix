#include <optix_helpers/Context.h>

namespace optix_helpers {

ContextObj::ContextObj() :
    context_(optix::Context::create())
{
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

Buffer ContextObj::create_buffer(RTbuffertype bufferType, const std::string& name) const
{
    return Buffer(new BufferObj(context_->createBuffer(bufferType), name));
}

RayType ContextObj::create_raytype(const Source& rayDefinition) const
{
    unsigned int rayTypeIndex = context_->getRayTypeCount();
    context_->setRayTypeCount(rayTypeIndex + 1);
    return RayType(new RayTypeObj(rayTypeIndex, rayDefinition));
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

//GeometryTriangles ContextObj::create_geometry_triangles() const
//{
//    return GeometryTriangles(context_->createGeometryTriangles(),
//                             context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3),
//                             context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3));
//}
//
//Model ContextObj::create_model() const
//{
//    return Model(context_->createGeometryInstance());
//}
//
//RayGenerator ContextObj::create_raygenerator(size_t width, size_t height, size_t depth) const
//{
//    return RayGenerator(width, height, depth, context_->createBuffer(RT_BUFFER_OUTPUT));
//}
//
//SceneItem ContextObj::create_scene_item(const Model& model, const char* acceleration) const
//{
//    return SceneItem(context_->createGeometryGroup(),
//                     context_->createTransform(),
//                     context_->createAcceleration(acceleration),
//                     model);
//}

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

Context create_context()
{
    return Context(new ContextObj());
}

} //namespace optix_helpers

