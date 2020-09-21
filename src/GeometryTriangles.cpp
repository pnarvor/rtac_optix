#include <optix_helpers/GeometryTriangles.h>

namespace optix_helpers {

GeometryTrianglesObj::GeometryTrianglesObj(const Context& context,
                                           bool withFaces,
                                           bool withNormals,
                                           bool withTextureCoordinates) :
    geometry_((*context)->createGeometryTriangles()),
    points_((*context)->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3))
{
    if(withFaces) {
        faces_ = (*context)->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3);
    }
    if(withNormals) {
        normals_ = (*context)->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3);
    }
    if(withTextureCoordinates) {
        textureCoordinates_ = (*context)->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2);
    }
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

