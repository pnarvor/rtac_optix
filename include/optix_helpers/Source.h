#ifndef _DEF_RTAC_OPTIX_SOURCE_H_
#define _DEF_RTAC_OPTIX_SOURCE_H_

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <vector>

#include <optix_helpers/Handle.h>

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

    int num_lines() const;
    std::ostream& print(std::ostream& os)  const;
};

class Source : public Handle<SourceObj>
{
    public:

    Source();
    Source(const std::string& source, const std::string& name);
};
using Sources = std::vector<Source>;

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Source& source);

#endif //_DEF_RTAC_OPTIX_SOURCE_H_
