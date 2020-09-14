#ifndef _DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_
#define _DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <rtac_base/types/Mesh.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Program.h>

namespace optix_helpers {

class GeometryTrianglesObj
{
    protected:

    optix::GeometryTriangles geometry_;
    optix::Buffer points_;
    optix::Buffer faces_;
    optix::Buffer normals_;
    optix::Buffer textureCoordinates_;

    public:

    GeometryTrianglesObj(const optix::GeometryTriangles& geometry,
                         const optix::Buffer& points,
                         const optix::Buffer& faces = optix::Buffer(),
                         const optix::Buffer& normals = optix::Buffer(),
                         const optix::Buffer& textureCoordinates = optix::Buffer());
    
    template <typename T>
    void set_points(size_t count, const T* points);
    template <typename T>
    void set_faces(size_t count, const T* faces);
    template <typename T>
    void set_normals(size_t count, const T* normals);
    template <typename T>
    void set_texture_coordinates(size_t count, const T* texCoords);
    template <typename Tp, typename Tf>
    void set_mesh(const rtac::types::Mesh<Tp,Tf,3>& mesh);

    optix::Buffer points()  const;
    optix::Buffer faces()   const;
    optix::Buffer normals() const;
    optix::Buffer texture_coordinates() const;

    size_t num_vertices() const;
    size_t num_faces() const;

    optix::GeometryTriangles geometry()   const;
    operator optix::GeometryTriangles()   const;
    optix::GeometryTriangles operator->() const;
};
using GeometryTriangles = Handle<GeometryTrianglesObj>;

template <typename T>
void GeometryTrianglesObj::set_points(size_t count, const T* points)
{
    points_->setSize(count);
    float* deviceData = static_cast<float*>(points_->map());
    for(int i = 0; i < 3*count; i++) {
        deviceData[i] = points[i];
    }
    points_->unmap();
    geometry_->setVertices(count, points_, points_->getFormat());
}

template <typename T>
void GeometryTrianglesObj::set_faces(size_t count, const T* faces)
{
    faces_->setSize(count);
    uint32_t* deviceFaces = static_cast<uint32_t*>(faces_->map());
    for(int i = 0; i < 3*count; i++) {
        deviceFaces[i] = faces[i];
    }
    faces_->unmap();
    geometry_->setPrimitiveCount(count);
    geometry_->setTriangleIndices(faces_, faces_->getFormat());
}

template <typename T>
void GeometryTrianglesObj::set_normals(size_t count, const T* normals)
{
    if(count != this->num_vertices()) {
        throw std::runtime_error(
            "GeometryTriangles : number of normals must be equal to number of vertices.");
    }
    normals_->setSize(count);
    float* deviceData = static_cast<float*>(normals_->map());
    for(int i = 0; i < 3*count; i++) {
        deviceData[i] = normals[i];
    }
    normals_->unmap();
}

template <typename T>
void GeometryTrianglesObj::set_texture_coordinates(size_t count, const T* texCoords)
{
    if(count != this->num_vertices()) {
        throw std::runtime_error(
            "GeometryTriangles : number of texCoords must be equal to number of vertices.");
    }
    textureCoordinates_->setSize(count);
    float* deviceData = static_cast<float*>(textureCoordinates_->map());
    for(int i = 0; i < 2*count; i++) {
        deviceData[i] = texCoords[i];
    }
    textureCoordinates_->unmap();
}


template <typename Tp, typename Tf>
void GeometryTrianglesObj::set_mesh(const rtac::types::Mesh<Tp,Tf,3>& mesh)
{
    // cannot use directly set_poitn and set_face because Eigen in column-wise order
    // change this.
    points_->setSize(mesh.num_points());
    auto points = mesh.points();
    float* devicePoints = static_cast<float*>(points_->map());
    for(int i = 0; i < mesh.num_points(); i++) {
        devicePoints[3*i]     = points(i,0);
        devicePoints[3*i + 1] = points(i,1);
        devicePoints[3*i + 2] = points(i,2);
    }
    points_->unmap();
    geometry_->setVertices(mesh.num_points(), points_, points_->getFormat());

    faces_->setSize(mesh.num_faces());
    auto faces = mesh.faces();
    uint32_t* deviceFaces = static_cast<uint32_t*>(faces_->map());
    for(int i = 0; i < mesh.num_faces(); i++) {
        deviceFaces[3*i]     = faces(i,0);
        deviceFaces[3*i + 1] = faces(i,1);
        deviceFaces[3*i + 2] = faces(i,2);
    }
    faces_->unmap();
    geometry_->setPrimitiveCount(mesh.num_faces());
    geometry_->setTriangleIndices(faces_, faces_->getFormat());
}

}; //namespace optix_helpers
#endif //_DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_
