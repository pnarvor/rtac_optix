#include <optix_helpers/GeometryTriangles.h>

namespace optix_helpers {

GeometryTrianglesObj::GeometryTrianglesObj(const optix::GeometryTriangles& geometry,
                                           const optix::Buffer& points,
                                           const optix::Buffer& faces,
                                           const optix::Buffer& normals,
                                           const optix::Buffer& textureCoordinates) :
    geometry_(geometry),
    points_(points),
    faces_(faces),
    normals_(normals),
    textureCoordinates_(textureCoordinates)
{
}

optix::Buffer GeometryTrianglesObj::points() const
{
    return points_;
}

optix::Buffer GeometryTrianglesObj::faces() const
{
    return faces_;
}

optix::Buffer GeometryTrianglesObj::normals() const
{
    return normals_;
}

optix::Buffer GeometryTrianglesObj::texture_coordinates() const
{
    return textureCoordinates_;
}

size_t GeometryTrianglesObj::num_vertices() const
{
    size_t count = 0;
    if(points_) {
        points_->getSize(count);
    }
    return count;
}

size_t GeometryTrianglesObj::num_faces() const
{
    size_t count = 0;
    if(faces_) {
        faces_->getSize(count);
    }
    return count;
}

optix::GeometryTriangles GeometryTrianglesObj::geometry() const
{
    return geometry_;
}

GeometryTrianglesObj::operator optix::GeometryTriangles() const
{
    return geometry_;
}

optix::GeometryTriangles GeometryTrianglesObj::operator->() const
{
    return geometry_;
}

}; //namespace optix_helpers

