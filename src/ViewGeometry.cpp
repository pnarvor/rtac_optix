#include <optix_helpers/ViewGeometry.h>

namespace optix_helpers {

ViewGeometryObj::ViewGeometryObj(const Source& definition) :
    definition_(definition)
{}

void ViewGeometryObj::set_callback_program(const RayGenerationProgram& program)
{
    program_ = program;
}

Source ViewGeometryObj::definition() const
{
    return definition_;
}

ViewGeometry::ViewGeometry() :
    Handle<ViewGeometryObj>()
{}

ViewGeometry::ViewGeometry(const Source& definition) :
    Handle<ViewGeometryObj>(new ViewGeometryObj(definition))
{}

}; //namespace optix_helpers
