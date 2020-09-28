#ifndef _DEF_OPTIX_HELPERS_HANDLE_H_
#define _DEF_OPTIX_HELPERS_HANDLE_H_

#include <memory>

// This is a generic pointer mean to be subclassed for each value typed class
// of a pointer-based api

namespace optix_helpers {

template <class T>
class Handle
{
    public:

    template<typename> friend class Handle;

     //one single reference to the base ptr to be able to change it if needed (boost?)
    template <typename BaseType> 
    using BasePtr = std::shared_ptr<BaseType>;

    using Ptr      = BasePtr<T>;
    using ConstPtr = BasePtr<const T>;

    protected:

    Ptr obj_;

    public:

    Handle(T* obj = NULL) : obj_(obj) {}
    Handle(const Ptr& obj) : obj_(obj) {}
    template <typename Tother>
    Handle(Handle<Tother>& other) :
        obj_(std::static_pointer_cast<T>(other.obj_))
    {}
    template <typename Tother>
    Handle(const Handle<Tother>& other) :
        obj_(std::static_pointer_cast<T>(other.obj_))
    {}
    template<class ...P>
    Handle(P... args) : obj_(new T(args...)) {}

    Ptr ptr()        { return obj_; }
    Ptr operator->() { return obj_; }
    T&  operator*()  { return *obj_; }
    T*  get()        { return obj_->get(); }

    ConstPtr ptr()        const { return obj_; }
    ConstPtr operator->() const { return obj_; }
    const T& operator*()  const { return *obj_; }
    const T* get()        const { return obj_->get(); }

    Handle<T> copy() const { return Handle(new T(*obj_)); }

    operator bool() const { return (bool)obj_; }
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_HANDLE_H_
