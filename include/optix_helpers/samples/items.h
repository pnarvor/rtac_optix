#ifndef _DEF_OPTIX_HELPERS_SAMPLES_ITEMS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_ITEMS_H_

#include <array>

#include <rtac_base/types/Mesh.h>

#include <optix_helpers/Context.h>
#include <optix_helpers/SceneItem.h>

#include <optix_helpers/samples/raytypes.h>
#include <optix_helpers/samples/geometries.h>
#include <optix_helpers/samples/materials.h>

namespace optix_helpers { namespace samples { namespace items {

SceneItem cube(const Context& context, const Materials& materials = Materials(),
               float scale = 1.0);
SceneItem sphere(const Context& context, const Materials& materials = Materials(),
                 float radius = 1.0);
SceneItem square(const Context& context, const Materials& materials = Materials(),
                 float scale = 1.0);
SceneItem tube(const Context& context, const Materials& materials = Materials(),
               float radius = 1.0, float height = 1.0);
SceneItem parabola(const Context& context, const Materials& materials = Materials(),
                   float a = 1.0, float b = 0.0, float height = 1.0);
template <typename Tp, typename Tf>
SceneItem mesh(const Context& context, const rtac::types::Mesh<Tp,Tf,3>& m,
               const Materials& materials = Materials());

// Implementation
template <typename Tp, typename Tf>
SceneItem mesh(const Context& context, const rtac::types::Mesh<Tp,Tf,3>& m,
               const Materials& materials)
{
    auto model = Model::New(context);
    model->set_geometry(geometries::mesh(context, m));
    for(auto material : materials) {
        model->add_material(material);
    }
    return SceneItem::New(context, model);
}

}; //namespace items
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_ITEMS_H_
