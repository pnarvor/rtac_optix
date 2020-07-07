#ifndef _DEF_OPTIX_HELPERS_NVRTC_H_
#define _DEF_OPTIX_HELPERS_NVRTC_H_

#include <iostream>
#include <vector>

namespace optix {

class NVRTC_Helper
{
    protected:

    std::vector<const char*> nvcrtOptions_;

    void parse_options();
    
    public:

    NVRTC_Helper();
};

}; //namespace optix

#endif //_DEF_OPTIX_HELPERS_NVRTC_H_
