#ifndef _DEF_RTAC_OPTIX_SOURCE_H_
#define _DEF_RTAC_OPTIX_SOURCE_H_

#include <iostream>
#include <memory>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <vector>

namespace optix_helpers {

class SourceObj
{
    protected:

    std::string source_; // source code
    std::string name_;   // source name
                         // (in case of header, must be the #include <tag>)

    public:

    SourceObj(const std::string& source, const std::string& name);
    SourceObj(const SourceObj& other);

    std::string source() const;
    std::string name()   const;

    const char* source_str() const;
    const char* name_str() const;

    int num_lines() const;
};
using Source = std::shared_ptr<SourceObj>;
using Sources = std::vector<Source>;

Source create_source(const std::string& source, const std::string& name);

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Source& source);

#endif //_DEF_RTAC_OPTIX_SOURCE_H_
