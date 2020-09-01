#ifndef _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
#define _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_

#include <iostream>
#include <initializer_list>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Program.h>

namespace optix_helpers {

class RayGeneratorObj
{
    public:

    class Shape
    {
        public:

        size_t width;
        size_t height;
        size_t depth;

        public:

        Shape(size_t w = 1, size_t h = 1, size_t d = 1);
        Shape(const std::initializer_list<size_t>& dims);

        size_t size() const;
    };

    protected:
    
    Shape                shape_;
    optix::Buffer        renderBuffer_; //buffer where the image will be rendered.
    RayGenerationProgram raygenProgram_;
    Program              missProgram_;

    void update_buffer_size();
    void update_buffer_variable();

    public:

    RayGeneratorObj(size_t width, size_t height, size_t depth,
                    const optix::Buffer& buffer,
                    const RayGenerationProgram& raygen = RayGenerationProgram(),
                    const Program& miss = Program());

    void set_raygen_program(const RayGenerationProgram& program);
    void set_miss_program(const Program& program);

    optix::Buffer render_buffer() const;
    RayGenerationProgram raygen_program() const;
    Program miss_program() const;
};

class RayGenerator : public Handle<RayGeneratorObj>
{
    public:

    RayGenerator();
    RayGenerator(size_t width, size_t height, size_t depth,
                 const optix::Buffer& buffer,
                 const RayGenerationProgram& raygen = RayGenerationProgram(),
                 const Program& miss = Program());
};


}; //namespace optix helpers


#endif //_DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
