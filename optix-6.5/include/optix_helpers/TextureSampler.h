#ifndef _DEF_OPTIX_HELPERS_TEXTURE_SAMPLER_H_
#define _DEF_OPTIX_HELPERS_TEXTURE_SAMPLER_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/NamedObject.h>
#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>

namespace optix_helpers {

class TextureSampler : public NamedObject<optix::TextureSampler>
{
    public:

    using Ptr      = Handle<TextureSampler>;
    using ConstPtr = Handle<const TextureSampler>;

    protected:

    void load_default_texture_setup();

    public:
    
    static Ptr New(const Context::ConstPtr& context, const std::string& name,
                   bool defaultSetup = true);
    TextureSampler(const Context::ConstPtr& context, const std::string& name,
                   bool defaultSetup = true);

    optix::TextureSampler       texture();
    const optix::TextureSampler texture() const;
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_TEXTURE_SAMPLER_H_
