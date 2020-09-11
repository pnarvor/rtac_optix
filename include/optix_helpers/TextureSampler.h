#ifndef _DEF_OPTIX_HELPERS_TEXTURE_SAMPLER_H_
#define _DEF_OPTIX_HELPERS_TEXTURE_SAMPLER_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/NamedObject.h>
#include <optix_helpers/Handle.h>

namespace optix_helpers {

class TextureSamplerObj : public NamedObject<optix::TextureSampler>
{
    public:
    
    TextureSamplerObj(const optix::TextureSampler& texture, const std::string& name);

    optix::TextureSampler       texture();
    const optix::TextureSampler texture() const;
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_TEXTURE_SAMPLER_H_
