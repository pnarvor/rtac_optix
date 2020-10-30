#ifndef _DEF_OPTIX_HELPERS_SAMPLES_UTILS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_UTILS_H_

#include <iostream>

#include <rtac_base/files.h>

#include <optix_helpers/Buffer.h>

namespace optix_helpers { namespace samples { namespace utils {

std::string to_file(const Buffer::ConstPtr& buffer, const std::string& path,
                    float a = 1.0, float b = 0.0);

void display(const Buffer::ConstPtr& buffer, float a = 1.0, float b = 0.0,
             const std::string& filePath = "tmp");
void display_ascii(const Buffer::ConstPtr& buffer, float a = 1.0, float b = 0.0,
                   std::ostream& os = std::cout, size_t maxWidth = 60);

}; //namespace utils
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_UTILS_H_

