#include <optix_helpers/utils.h>

namespace optix_helpers {

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

}; //namespace optix_helpers

