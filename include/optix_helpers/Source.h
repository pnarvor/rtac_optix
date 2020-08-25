#ifndef _DEF_RTAC_OPTIX_SOURCE_H_
#define _DEF_RTAC_OPTIX_SOURCE_H_

#include <iostream>
#include <memory>
#include <vector>

namespace optix_helpers {

class Source
{
    protected:

    std::string source_; // source code
    std::string name_;   // source name
                         // (in case of header, must be the #include <tag>)

    public:

    Source(const std::string& source = "", const std::string& name = "default_program");

    std::string source() const;
    std::string name()   const;
};
using Sources = std::vector<Source>;

}; //namespace optix_helpers

#endif //_DEF_RTAC_OPTIX_SOURCE_H_
