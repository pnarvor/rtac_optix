#ifndef _DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_
#define _DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_

#include <iostream>
#include <memory>

#include <optixu/optixpp.h>

#include <optix_helpers/Program.h>

namespace optix_helpers {

class GeometryTrianglesObj
{
    protected:

    optix::GeometryTriangles geometry_;
    optix::Buffer points_;
    optix::Buffer faces_;

    public:

    GeometryTrianglesObj(const optix::GeometryTriangles& geometry,
                         const optix::Buffer& points,
                         const optix::Buffer& faces);
    
    template <typename T>
    void set_points(size_t numPoints, const T* points);
    template <typename T>
    void set_faces(size_t numFaces, const T* faces);

    optix::Buffer points() const;
    optix::Buffer faces() const;
    optix::GeometryTriangles geometry()   const;
    operator optix::GeometryTriangles()   const;
    optix::GeometryTriangles operator->() const;
};
using GeometryTriangles = std::shared_ptr<GeometryTrianglesObj>;

template <typename T>
void GeometryTrianglesObj::set_points(size_t numPoints, const T* points)
{
    points_->setSize(numPoints);
    float* devicePoints = static_cast<float*>(points_->map());
    for(int i = 0; i < 3*numPoints; i++) {
        devicePoints[i] = points[i];
    }
    points_->unmap();
    geometry_->setVertices(numPoints, points_, points_->getFormat());
}

template <typename T>
void GeometryTrianglesObj::set_faces(size_t numFaces, const T* faces)
{
    faces_->setSize(numFaces);
    uint32_t* deviceFaces = static_cast<uint32_t*>(faces_->map());
    for(int i = 0; i < 3*numFaces; i++) {
        deviceFaces[i] = faces[i];
    }
    faces_->unmap();
    geometry_->setPrimitiveCount(numFaces);
    geometry_->setTriangleIndices(faces_, faces_->getFormat());
}

}; //namespace optix_helpers
#endif //_DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_
