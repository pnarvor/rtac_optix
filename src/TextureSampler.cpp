#include <optix_helpers/TextureSampler.h>

namespace optix_helpers {

TextureSamplerObj::TextureSamplerObj(const optix::TextureSampler& texture, const std::string& name) :
    NamedObject<optix::TextureSampler>(texture, name)
{}

optix::TextureSampler TextureSamplerObj::texture()
{
    return object_;
}

const optix::TextureSampler TextureSamplerObj::texture() const
{
    return object_;
}

}; //namespace optix_helpers
