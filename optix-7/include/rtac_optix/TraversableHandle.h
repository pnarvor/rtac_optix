#ifndef _DEF_RTAC_OPTIX_TRAVERSABLE_HANDLE_H_
#define _DEF_RTAC_OPTIX_TRAVERSABLE_HANDLE_H_

#include <iostream>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>

namespace rtac { namespace optix {

struct TraversableHandle
{
    // ABSTRACT class

    // This is an abstract class to represent the base type for nodes in the
    // rendering tree. The main purpose of this type is to be able to infer the
    // sbt records positions by exploring the world tree.

    // This class represent a node of the scene graph in which the ray will
    // propagate. OptixTraversableHandle is a generic type which can represent
    // any item in the graph (geometries, transforms, "structural" nodes
    // containing other nodes, etc...)
    
    public:

    using Ptr      = Handle<TraversableHandle>;
    using ConstPtr = Handle<const TraversableHandle>;
    
    virtual operator OptixTraversableHandle() = 0;

    // This represent the width in sbt offsets that the 
    virtual unsigned int sbt_width() const = 0;
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_TRAVERSABLE_HANDLE_H_
