#include <optix_helpers/samples/utils.h>

namespace optix_helpers { namespace samples { namespace utils {

std::string to_file(const Buffer& buffer, const std::string& path, float a, float b)
{
    std::string realPath = path;
    size_t W, H;
    (*buffer)->getSize(W,H);

    switch((*buffer)->getFormat()) {
        case RT_FORMAT_FLOAT:
            realPath = rtac::files::append_extension(path, "pgm");
            rtac::files::write_pgm(realPath, W, H,
                static_cast<const float*>((*buffer)->map()));
            (*buffer)->unmap();
            break;
        case RT_FORMAT_FLOAT3:
            realPath = rtac::files::append_extension(path, "ppm");
            rtac::files::write_ppm(realPath, W, H,
                static_cast<const float*>((*buffer)->map()));
            (*buffer)->unmap();
            break;
        default:
            throw std::runtime_error("Wrong format for ppm/pgm file export.");
    }
    return realPath;
}

void display(const Buffer& buffer, float a, float b, const std::string& filePath)
{
    system((std::string("eog ") + to_file(buffer, filePath, a ,b)).c_str());
}

void display_ascii(const Buffer& buffer, float a, float b, 
                   std::ostream& os, size_t maxWidth)
{
    size_t W, H;
    (*buffer)->getSize(W,H);
    int resampleFactor = 1;
    for(; W / resampleFactor > maxWidth; resampleFactor++);

    const float* data = static_cast<const float*>((*buffer)->map());
    switch((*buffer)->getFormat()) {
        case RT_FORMAT_FLOAT:
            for(int h = resampleFactor / 2; h < H; h += resampleFactor) {
                for(int w = resampleFactor / 2; w < W; w += resampleFactor) {
                    int index = (W*h + w);
                    os << (int)(a*data[index] + b) << " ";
                }
                os << "\n";
            }
            break;
        case RT_FORMAT_FLOAT3:
            for(int h = resampleFactor / 2; h < H; h += resampleFactor) {
                for(int w = resampleFactor / 2; w < W; w += resampleFactor) {
                    int index = 3*(W*h + w);
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

