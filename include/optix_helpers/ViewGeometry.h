#ifndef _DEF_OPTIX_HELPERS_VIEW_GEOMETRY_H_
#define _DEF_OPTIX_HELPERS_VIEW_GEOMETRY_H_

#include <iostream>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>
#include <optix_helpers/Program.h>

namespace optix_helpers {

// Defines geometrical ray parameters (origin, direction, ...)
class ViewGeometryObj
{
    protected:

    Source definition_; // header to be used in a RayGeneration program.
    RayGenerationProgram program_; // program to be updated with 

    public:

    ViewGeometryObj(const Source& definition);
    
    void set_callback_program(const RayGenerationProgram& program);

    Source definition() const;
};

class ViewGeometry : public Handle<ViewGeometryObj>
{
    public:

    ViewGeometry();
    ViewGeometry(const Source& definition);
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_VIEW_GEOMETRY_H_


