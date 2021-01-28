#ifndef _DEF_OPTIX_HELPERS_HANDLE_H_
#define _DEF_OPTIX_HELPERS_HANDLE_H_

#include <rtac_base/types/Handle.h>

namespace optix_helpers {

template <typename T>
using Handle = rtac::types::Handle<T>; // Is actually a std::shared_ptr

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_HANDLE_H_
