#include <optix_helpers/Context.h>

namespace optix_helpers {

Context::Context() :
    context_(optix::Context::create())
{
}

Program Context::create_program(const Source& source, const Sources& additionalHeaders) const
{
    try {
        auto ptx = nvrtc_.compile(source, additionalHeaders);
        optix::Program program = context_->createProgramFromPTXString(ptx, source.name());
        return Program(source, additionalHeaders, program);
    }
    catch(const std::runtime_error& e) {
        std::ostringstream os;
        os << source <<  "\n" << e.what();
        throw std::runtime_error(os.str());
    }
}

Buffer Context::create_buffer(RTbuffertype bufferType, const std::string& name) const
{
    return Buffer(context_->createBuffer(bufferType), name);
}

RayType Context::create_raytype(const Source& rayDefinition) const
{
    unsigned int rayTypeIndex = context_->getRayTypeCount();
    context_->setRayTypeCount(rayTypeIndex + 1);
    return RayType(rayTypeIndex, rayDefinition);
}

//Material Context::create_material() const
//{
//    return Material(this->context()->createMaterial());
//}
//
//Geometry Context::create_geometry(const Program& intersection,
//                                     const Program& boundingbox,
//                                     size_t primitiveCount) const
//{
//    return Geometry(context_->createGeometry(), intersection, boundingbox, primitiveCount);
//}
//
//GeometryTriangles Context::create_geometry_triangles() const
//{
//    return GeometryTriangles(context_->createGeometryTriangles(),
//                             context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3),
//                             context_->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3));
//}
//
//Model Context::create_model() const
//{
//    return Model(context_->createGeometryInstance());
//}
//
//RayGenerator Context::create_raygenerator(size_t width, size_t height, size_t depth) const
//{
//    return RayGenerator(width, height, depth, context_->createBuffer(RT_BUFFER_OUTPUT));
//}
//
//SceneItem Context::create_scene_item(const Model& model, const char* acceleration) const
//{
//    return SceneItem(context_->createGeometryGroup(),
//                     context_->createTransform(),
//                     context_->createAcceleration(acceleration),
//                     model);
//}

optix::Handle<optix::VariableObj> Context::operator[](const std::string& varname)
{
    return context_[varname];
}

Context::operator optix::Context() const
{
    return context_;
}

optix::Context Context::operator->()
{
    return context_;
}

optix::Context Context::operator->() const
{
    return context_;
}

optix::Context Context::context() const
{
    return context_;
}

} //namespace optix_helpers

