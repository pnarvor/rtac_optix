#ifndef _DEF_OPTIX_HELPERS_SAMPLES_TEST_SCENES_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_TEST_SCENES_H_

#include <iostream>

#include <rtac_base/types/Pose.h>

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/PinHoleView.h>

#include <optix_helpers/samples/raytypes.h>
#include <optix_helpers/samples/materials.h>
#include <optix_helpers/samples/geometries.h>
#include <optix_helpers/samples/items.h>
#include <optix_helpers/samples/utils.h>

namespace optix_helpers { namespace samples { namespace scenes {

class Scene0
{
    public:

    using Pose = rtac::types::Pose<float>;
    using Quaternion = rtac::types::Quaternion<float>;

    static const Source raygenSource;

    Context context_;
    PinHoleView view_;

    public:

    Scene0(size_t width, size_t height,
           unsigned int glboId = 0);

    ViewGeometry view();
    void launch();
};


}; //namespace scenes
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_TEST_SCENES_H_
