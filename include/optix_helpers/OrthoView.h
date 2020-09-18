#ifndef _DEF_OPTIX_HELPERS_ORTHO_VIEW_H_
#define _DEF_OPTIX_HELPERS_ORTHO_VIEW_H_

#include <iostream>
#include <initializer_list>

#include <rtac_base/types/common.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>
#include <optix_helpers/ViewGeometry.h>

namespace optix_helpers {

class OrthoViewObj : public ViewGeometryObj
{
    public:

    static const Source rayGeometryDefinition;

    struct Bounds
    {
        float left;
        float right;
        float bottom;
        float top;
    };
    using Pose = ViewGeometry::Pose;
    using Vector3 = ViewGeometry::Vector3;
    using Matrix3 = ViewGeometry::Matrix3;

    protected:

    Bounds bounds_;

    void update_device_geometry();

    public:

    OrthoViewObj(const Buffer& renderBuffer,
                 const Program& raygenProgram,
                 const Bounds& bounds = Bounds({-1.0,1.0,-1.0,1.0}),
                 const Pose& pose = Pose());
    
    virtual void set_pose(const Pose& pose);
    virtual void set_size(size_t width, size_t height);
    virtual void set_range(float zNear, float zFar);

    void set_bounds(const Bounds& bounds);
};

class OrthoView : public Handle<OrthoViewObj>
{
    public:
    
    using Pose    = OrthoViewObj::Pose;
    using Bounds  = OrthoViewObj::Bounds;
    using Vector3 = OrthoViewObj::Vector3;
    using Matrix3 = OrthoViewObj::Matrix3;
    static const Source& rayGeometryDefinition;
    
    OrthoView();
    OrthoView(const Buffer& renderBuffer,
              const Program& raygenProgram,
              const Bounds& bounds = Bounds({-1.0,1.0,-1.0,1.0}),
              const Pose& pose = Pose());
};

};

#endif //_DEF_OPTIX_HELPERS_ORTHO_VIEW_H_
