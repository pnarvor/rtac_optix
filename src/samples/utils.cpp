#include <optix_helpers/samples/utils.h>

namespace optix_helpers { namespace samples { namespace utils {

std::string to_file(const Buffer::ConstPtr& buffer, const std::string& path, float a, float b)
{
    std::string realPath = path;
    auto shape = buffer->shape();

    switch((*buffer)->getFormat()) {
        case RT_FORMAT_FLOAT:
            realPath = rtac::files::append_extension(path, "pgm");
            rtac::files::write_pgm(realPath, shape.width, shape.height,
                buffer->map<const float*>());
            buffer->unmap();
            break;
        case RT_FORMAT_FLOAT3:
            realPath = rtac::files::append_extension(path, "ppm");
            rtac::files::write_ppm(realPath, shape.width, shape.height,
                buffer->map<const float*>());
            buffer->unmap();
            break;
        default:
            throw std::runtime_error("Wrong format for ppm/pgm file export.");
    }
    return realPath;
}

void display(const Buffer::ConstPtr& buffer, float a, float b, const std::string& filePath)
{
    system((std::string("eog ") + to_file(buffer, filePath, a ,b)).c_str());
}

void display_ascii(const Buffer::ConstPtr& buffer, float a, float b, 
                   std::ostream& os, size_t maxWidth)
{
    auto shape = buffer->shape();
    int resampleFactor = 1;
    for(; shape.width / resampleFactor > maxWidth; resampleFactor++);

    auto data = buffer->map<const float*>();
    switch((*buffer)->getFormat()) {
        case RT_FORMAT_FLOAT:
            for(int h = resampleFactor / 2; h < shape.height; h += resampleFactor) {
                for(int w = resampleFactor / 2; w < shape.width; w += resampleFactor) {
                    int index = (shape.width*h + w);
                    os << (int)(a*data[index] + b) << " ";
                }
                os << "\n";
            }
            break;
        case RT_FORMAT_FLOAT3:
            for(int h = resampleFactor / 2; h < shape.height; h += resampleFactor) {
                for(int w = resampleFactor / 2; w < shape.width; w += resampleFactor) {
                    int index = 3*(shape.width*h + w);
                    os << (int)(a*data[index] + b) << " ";
                }
                os << "\n";
            }
            break;
        default:
            throw std::runtime_error("Wrong format for ascii display.");
    }
    (*buffer)->unmap();
}

}; //namespace utils
}; //namespace samples
}; //namespace optix_helpers

