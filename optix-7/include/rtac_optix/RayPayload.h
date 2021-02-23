#ifndef _DEF_RTAC_OPTIX_RAY_PAYLOAD_H_
#define _DEF_RTAC_OPTIX_RAY_PAYLOAD_H_

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_base/cuda/utils.h>

namespace rtac { namespace optix {

template <class T>
struct RayPayload : public T
{
    // This struct is intended to abstract/simplify/optimize the use of optix
    // payload registers by providing compile time optimized helpers.

    public:

    //RayPayload() : T() {}
    //RayPayload(const T& payload) : T(payload) {}
    
    // Mininal number of 4 bytes registers to hold T type data.
    static constexpr unsigned int RegisterCount = (sizeof(T) + 3) / 4; 
    static constexpr unsigned int PaddingSize   = 4*RegisterCount - sizeof(T);
    
    // This padding is to ensure this struct size is aligned on uint32_t
    // boundaries.  This allows a generic copy to/from optix ray payload
    // registers.
    uint8_t pad__[PaddingSize];

    // Compile time check if no more than 8 payload registers are used.
    static_assert(RegisterCount <= 8,
                  "Payload size too big for number of ray payload registers (max 8x4 bytes). If your payload is supposed to be small enough, check alignment");


    #ifdef RTAC_CUDACC // Register getter/setters
    
    // return pointer to this cast to uint32_t
    __device__ const uint32_t* data() const;
    __device__       uint32_t* data();
    
    // copy from/to optix ray payload registers using optixSetPayload_n and optixGetPayload_n
    __device__ void to_registers() const;
    __device__ static RayPayload<T> from_registers();
    
    #endif //RTAC_CUDACC
};


#ifdef RTAC_CUDACC // Register getter/setters

template <class T> __device__
const uint32_t* RayPayload<T>::data() const
{
    return reinterpret_cast<const uint32_t*>(this);
}

template <class T> __device__
uint32_t* RayPayload<T>::data()
{
    return reinterpret_cast<uint32_t*>(this);
}

template <class T> __device__
void RayPayload<T>::to_registers() const
{
    // if constexpr is a c++17 feature which allows to ignore code at compile
    // time by checking the result of a constexpr test.
    if constexpr(RegisterCount > 0) optixSetPayload_0(this->data()[0]);
    if constexpr(RegisterCount > 1) optixSetPayload_1(this->data()[1]);
    if constexpr(RegisterCount > 2) optixSetPayload_2(this->data()[2]);
    if constexpr(RegisterCount > 3) optixSetPayload_3(this->data()[3]);
    if constexpr(RegisterCount > 4) optixSetPayload_4(this->data()[4]);
    if constexpr(RegisterCount > 5) optixSetPayload_5(this->data()[5]);
    if constexpr(RegisterCount > 6) optixSetPayload_6(this->data()[6]);
    if constexpr(RegisterCount > 7) optixSetPayload_7(this->data()[7]);
}

template <class T> __device__
RayPayload<T> RayPayload<T>::from_registers()
{
    RayPayload<T> res;

    // if constexpr is a c++17 feature which allows to ignore code at compile
    // time by checking the result of a constexpr test.
    if constexpr(RegisterCount > 0) res.data()[0] = optixGetPayload_0();
    if constexpr(RegisterCount > 1) res.data()[1] = optixGetPayload_1();
    if constexpr(RegisterCount > 2) res.data()[2] = optixGetPayload_2();
    if constexpr(RegisterCount > 3) res.data()[3] = optixGetPayload_3();
    if constexpr(RegisterCount > 4) res.data()[4] = optixGetPayload_4();
    if constexpr(RegisterCount > 5) res.data()[5] = optixGetPayload_5();
    if constexpr(RegisterCount > 6) res.data()[6] = optixGetPayload_6();
    if constexpr(RegisterCount > 7) res.data()[7] = optixGetPayload_7();

    return res;
}

#endif //RTAC_CUDACC

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_RAY_PAYLOAD_H_
