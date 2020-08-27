#ifndef _DEF_OPTIX_HELPERS_HANDLE_H_
#define _DEF_OPTIX_HELPERS_HANDLE_H_

#include <memory>

// This is a generic pointer mean to be subclassed for each value typed class
// of a pointer-based api

template <class T>
class Handle
{
    protected:

    std::shared_ptr<T> obj_;

    public:

    Handle(T* obj = NULL) : obj_(obj) {}

    std::shared_ptr<T> operator->() { return obj_; }
    T& operator*() { return *obj_; }
    T* get() { return obj_->get(); }

    std::shared_ptr<const T> operator->() const { return obj_; }
    const T& operator*() const { return *obj_; }
    const T* get() const { return obj_->get(); }

    Handle<T> copy() const { return Handle(new T(*obj_)); }

    operator bool() const { return (bool)obj_; }
};

#endif //_DEF_OPTIX_HELPERS_HANDLE_H_
