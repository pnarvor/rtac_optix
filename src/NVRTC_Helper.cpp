#include <optix_helpers/NVRTC_Helper.h>
#include <list>

namespace optix {

void NVRTC_Helper::parse_options()
{
    // parse include directories and compile option from NVRTC_INCLUDE_DIRS
    // definitions.

    // parsing include dirs
    std::string includeDirs(&NVRTC_INCLUDE_DIRS[1]);
    // removing unwanted quotes (results from setting the definition through
    // cmake. See if better way)
    if(includeDirs[0] == '\"')
        includeDirs.erase(0,1);
    if(includeDirs[includeDirs.size()-1] == '\"')
        includeDirs.pop_back();

    //includeDirs is a semi-colon separated directory list. Let's split.
    std::list<std::string> dirList;
    size_t pos;
    while((pos = includeDirs.find(";")) != std::string::npos) {
        dirList.push_back(includeDirs.substr(0, pos));
        includeDirs.erase(0, pos + 1);
    }
    dirList.push_back(includeDirs);
    for(auto v : dirList) {
        std::cout << v << std::endl;
    }
}

NVRTC_Helper::NVRTC_Helper()
{
    this->parse_options();
}



}; //namespace optix

