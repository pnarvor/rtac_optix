#include <optix_helpers/TextureSampler.h>

namespace optix_helpers {

TextureSamplerObj::TextureSamplerObj(const Context& context, 
                                     const std::string& name,
                                     bool defaultSetup) :
    NamedObject<optix::TextureSampler>((*context)->createTextureSampler(), name)
{
    if(defaultSetup) {
        this->load_default_texture_setup();
    }
}

void TextureSamplerObj::load_default_texture_setup()
{
    object_->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    object_->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    object_->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    object_->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
    object_->setReadMode(RT_TEXTURE_READ_NORMALIZED_FLOAT);
    object_->setMaxAnisotropy(1.f);
}

optix::TextureSampler TextureSamplerObj::texture()
{
    return object_;
}

const optix::TextureSampler TextureSamplerObj::texture() const
{
    return object_;
}

}; //namespace optix_helpers
