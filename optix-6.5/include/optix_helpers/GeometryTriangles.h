#ifndef _DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_
#define _DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <rtac_base/types/Mesh.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Program.h>
#include <optix_helpers/Buffer.h>

namespace optix_helpers {

class GeometryTriangles
{
    public:

    using Ptr      = Handle<GeometryTriangles>;
    using ConstPtr = Handle<const GeometryTriangles>;
    
    template <typename Tp, typename Tf>
    using Mesh = rtac::types::Mesh<Tp,Tf,3>;

    protected:

    optix::GeometryTriangles geometry_;
    Buffer::Ptr points_;
    Buffer::Ptr faces_;
    Buffer::Ptr normals_;
    Buffer::Ptr textureCoordinates_;

    public:

    static Ptr New(const Context::ConstPtr& context,
                   bool withFaces = true,
                   bool withNormals = false,
                   bool withTextureCoordinates = false);
    GeometryTriangles(const Context::ConstPtr& context,
                      bool withFaces = true,
                      bool withNormals = false,
                      bool withTextureCoordinates = false);
    template <typename Tp, typename Tf>
    GeometryTriangles(const Context::ConstPtr& context, const Mesh<Tp,Tf>& mesh);
    
    template <typename T>
    void set_points(size_t count, const T* points);
    template <typename T>
    void set_faces(size_t count, const T* faces);
    template <typename T>
    void set_normals(size_t count, const T* normals);
    template <typename T>
    void set_texture_coordinates(size_t count, const T* texCoords);
    template <typename Tp, typename Tf>
    void set_mesh(const Mesh<Tp,Tf>& mesh);

    Buffer::Ptr points();
    Buffer::Ptr faces();
    Buffer::Ptr normals();
    Buffer::Ptr texture_coordinates();

    Buffer::ConstPtr points()  const;
    Buffer::ConstPtr faces()   const;
    Buffer::ConstPtr normals() const;
    Buffer::ConstPtr texture_coordinates() const;

    size_t num_vertices() const;
    size_t num_faces() const;

    optix::GeometryTriangles geometry()   const;
    operator optix::GeometryTriangles()   const;
    optix::GeometryTriangles operator->() const;
};

template <typename Tp, typename Tf>
GeometryTriangles::GeometryTriangles(const Context::ConstPtr& context, const Mesh<Tp,Tf>& mesh) :
    GeometryTriangles(context, true, false, false)
{
    this->set_mesh(mesh);
}

template <typename T>
void GeometryTriangles::set_points(size_t count, const T* points)
{
    points_->set_size(count);
    auto deviceData = points_->map<float*>();
    for(int i = 0; i < 3*count; i++) {
        deviceData[i] = points[i];
    }
    points_->unmap();
    geometry_->setVertices(count, *points_, (*points_)->getFormat());
}

template <typename T>
void GeometryTriangles::set_faces(size_t count, const T* faces)
{
    faces_->set_size(count);
    auto deviceFaces = faces_->map<uint32_t*>();
    for(int i = 0; i < 3*count; i++) {
        deviceFaces[i] = faces[i];
    }
    faces_->unmap();
    geometry_->setPrimitiveCount(count);
    geometry_->setTriangleIndices(*faces_, (*faces_)->getFormat());
}

template <typename T>
void GeometryTriangles::set_normals(size_t count, const T* normals)
{
    if(count != this->num_vertices()) {
        throw std::runtime_error(
            "GeometryTriangles : number of normals must be equal to number of vertices.");
    }
    normals_->set_size(count);
    auto deviceData = normals_->map<float*>();
    for(int i = 0; i < 3*count; i++) {
        deviceData[i] = normals[i];
    }
    normals_->unmap();
}

template <typename T>
void GeometryTriangles::set_texture_coordinates(size_t count, const T* texCoords)
{
    if(count != this->num_vertices()) {
        throw std::runtime_error(
            "GeometryTriangles : number of texCoords must be equal to number of vertices.");
    }
    textureCoordinates_->set_size(count);
    auto deviceData = textureCoordinates_->map<float*>();
    for(int i = 0; i < 2*count; i++) {
        deviceData[i] = texCoords[i];
    }
    textureCoordinates_->unmap();
}


template <typename Tp, typename Tf>
void GeometryTriangles::set_mesh(const Mesh<Tp,Tf>& mesh)
{
    // cannot use directly set_poitn and set_face because Eigen in column-wise order
    // change this.
    points_->set_size(mesh.num_points());
    auto points = mesh.points();
    auto devicePoints = points_->map<float*>();
    for(int i = 0; i < mesh.num_points(); i++) {
        devicePoints[3*i]     = points(i,0);
        devicePoints[3*i + 1] = points(i,1);
        devicePoints[3*i + 2] = points(i,2);
    }
    points_->unmap();
    geometry_->setVertices(mesh.num_points(), *points_, (*points_)->getFormat());

    faces_->set_size(mesh.num_faces());
    auto faces = mesh.faces();
    auto deviceFaces = faces_->map<uint32_t*>();
    for(int i = 0; i < mesh.num_faces(); i++) {
        deviceFaces[3*i]     = faces(i,0);
        deviceFaces[3*i + 1] = faces(i,1);
        deviceFaces[3*i + 2] = faces(i,2);
    }
    faces_->unmap();
    geometry_->setPrimitiveCount(mesh.num_faces());
    geometry_->setTriangleIndices(*faces_, (*faces_)->getFormat());
}

}; //namespace optix_helpers
#endif //_DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_
