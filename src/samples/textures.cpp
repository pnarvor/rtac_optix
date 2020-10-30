#include <optix_helpers/samples/textures.h>

namespace optix_helpers { namespace samples { namespace textures {

TextureSampler::Ptr checkerboard(const Context::ConstPtr& context,
                                 const std::string& textureName,
                                 const std::array<uint8_t,3>& color1,
                                 const std::array<uint8_t,3>& color2,
                                 size_t width, size_t height)
{
    auto buffer = Buffer::New(context, RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE4,
                  "checkerboard_data");
    buffer->set_size(width, height);
    auto data = buffer->map<uint8_t*>();
    for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w ++) {
            int i = 4*(width*h + w);
            if((h+w) % 2 == 0) {
                data[i]     = color1[0];
                data[i + 1] = color1[1];
                data[i + 2] = color1[2];
                data[i + 3] = 1.0;
            }
            else {
                data[i]     = color2[0];
                data[i + 1] = color2[1];
                data[i + 2] = color2[2];
                data[i + 3] = 1.0;
            }
        }
    }
    buffer->unmap();

    auto texture = TextureSampler::New(context, textureName);
    (*texture)->setBuffer(*buffer);
    
    // Behavior for texture coordinates outside border
    //(*texture)->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE); // repeat edge pixel.
    //(*texture)->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    //(*texture)->setWrapMode(0, RT_WRAP_MIRROR); //mirrored texture
    //(*texture)->setWrapMode(1, RT_WRAP_MIRROR);
    (*texture)->setWrapMode(0, RT_WRAP_REPEAT); //periodic texture
    (*texture)->setWrapMode(1, RT_WRAP_REPEAT);

    //(*texture)->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_NONE);
    (*texture)->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NONE);

    return texture;
}


}; //namespace textures
}; //namespace samples
}; //namespace optix_helpers

