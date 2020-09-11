#ifndef _DEF_OPTIX_HELPERS_NAMED_OBJ_H_
#define _DEF_OPTIX_HELPERS_NAMED_OBJ_H_

#include <iostream>

namespace optix_helpers {

// T should be a handle-like type
template <typename T>
class NamedObject
{
    protected:

    T object_;
    std::string name_;

    public:
    
    NamedObject(const T& object, const std::string& name) :
        object_(object), name_(name)
    {}

    operator T()
    {
        return object_;
    }

    operator T() const
    {
        return object_;
    }

    T operator->()
    {
        return object_;
    }

    //const T operator->() const
    T operator->() const // investigate this
    {
        return object_;
    }

    std::string name() const
    {
        return name_;
    }
};

}; //namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_NAMED_OPTIX_OBJ_H_
