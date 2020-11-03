#ifndef _DEF_OPTIX_HELPERS_PINHOLE_VIEW_H_
#define _DEF_OPTIX_HELPERS_PINHOLE_VIEW_H_

#include <iostream>
#include <initializer_list>

#include <rtac_base/types/common.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>
#include <optix_helpers/RayGenerator.h>

namespace optix_helpers { namespace samples { namespace raygenerators {

class PinHole : public RayGenerator
{
    public:

    using Ptr      = Handle<PinHole>;
    using ConstPtr = Handle<const PinHole>;

    static const Source::Ptr rayGeometryDefinition;

    using Pose    = RayGenerator::Pose;
    using Vector3 = RayGenerator::Vector3;
    using Matrix3 = RayGenerator::Matrix3;

    constexpr static const float defaultFovy = 90.0f;

    protected:

    float fovy_;

    virtual void update_geometry();

    public:

    static Ptr New(const Context::ConstPtr& context, 
                   const Buffer::Ptr& renderBuffer,
                   const RayType& rayType,
                   const Source::Ptr& raygenSource,
                   const Sources& additionalHeaders = {});

    PinHole(const Context::ConstPtr& context, 
            const Buffer::Ptr& renderBuffer,
            const RayType& rayType,
            const Source::Ptr& raygenSource,
            const Sources& additionalHeaders = {});

    virtual void set_range(float zNear, float zFar);

    void set_fovy(float fovy);
};

}; //namespace raygenerators
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_ORTHO_VIEW_H_
