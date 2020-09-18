#ifndef _DEF_OPTIX_HELPERS_ORTHO_VIEW_H_
#define _DEF_OPTIX_HELPERS_ORTHO_VIEW_H_

#include <iostream>
#include <initializer_list>

#include <rtac_base/types/common.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>
#include <optix_helpers/ViewGeometry.h>

namespace optix_helpers {

class PinHoleViewObj : public ViewGeometryObj
{
    public:

    static const Source rayGeometryDefinition;

    using Pose = ViewGeometry::Pose;
    using Vector3 = ViewGeometry::Vector3;
    using Matrix3 = ViewGeometry::Matrix3;

    protected:

    float fovy_;

    void update_device_geometry();

    public:

    PinHoleViewObj(const Buffer& renderBuffer,
                   const Program& raygenProgram,
                   float fovy = 90.0f,
                   const Pose& pose = Pose());

    virtual void set_pose(const Pose& pose);
    virtual void set_size(size_t width, size_t height);
    virtual void set_range(float zNear, float zFar);

    void set_fovy(float fovy);
};

class PinHoleView : public Handle<PinHoleViewObj>
{
    public:
    
    using Pose    = PinHoleViewObj::Pose;
    using Vector3 = PinHoleViewObj::Vector3;
    using Matrix3 = PinHoleViewObj::Matrix3;
    static const Source& rayGeometryDefinition;
    
    PinHoleView();
    PinHoleView(const Buffer& renderBuffer,
                const Program& raygenProgram,
                float fovy = 90.0,
                const Pose& pose = Pose());

    operator ViewGeometry();
};

};

#endif //_DEF_OPTIX_HELPERS_ORTHO_VIEW_H_
