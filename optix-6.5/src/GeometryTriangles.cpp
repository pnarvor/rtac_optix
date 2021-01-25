#include <optix_helpers/GeometryTriangles.h>

namespace optix_helpers {

GeometryTriangles::Ptr GeometryTriangles::New(const Context::ConstPtr& context,
                                              bool withFaces,
                                              bool withNormals,
                                              bool withTextureCoordinates)
{
    return Ptr(new GeometryTriangles(context, withFaces, withNormals,
                                     withTextureCoordinates));
}

GeometryTriangles::GeometryTriangles(const Context::ConstPtr& context,
                                     bool withFaces,
                                     bool withNormals,
                                     bool withTextureCoordinates) :
    geometry_((*context)->createGeometryTriangles()),
    points_(Buffer::New(context, RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, "vertex_buffer")),
    faces_(NULL),
    normals_(NULL),
    textureCoordinates_(NULL)
{
    if(withFaces) {
        faces_ = Buffer::New(context, RT_BUFFER_INPUT,
                             RT_FORMAT_UNSIGNED_INT3, "index_buffer");
    }
    if(withNormals) {
        normals_ = Buffer::New(context, RT_BUFFER_INPUT,
                               RT_FORMAT_FLOAT3, "normal_buffer");
    }
    if(withTextureCoordinates) {
        textureCoordinates_ = Buffer::New(context, RT_BUFFER_INPUT,
                                          RT_FORMAT_FLOAT2, "texcoord_buffer");
    }
}

Buffer::Ptr GeometryTriangles::points()
{
    return points_;
}

Buffer::Ptr GeometryTriangles::faces()
{
    return faces_;
}

Buffer::Ptr GeometryTriangles::normals()
{
    return normals_;
}

Buffer::Ptr GeometryTriangles::texture_coordinates()
{
    return textureCoordinates_;
}

Buffer::ConstPtr GeometryTriangles::points() const
{
    return points_;
}

Buffer::ConstPtr GeometryTriangles::faces() const
{
    return faces_;
}

Buffer::ConstPtr GeometryTriangles::normals() const
{
    return normals_;
}

Buffer::ConstPtr GeometryTriangles::texture_coordinates() const
{
    return textureCoordinates_;
}

size_t GeometryTriangles::num_vertices() const
{
    if(points_) {
        return points_->size();
    }
    return 0;
}

size_t GeometryTriangles::num_faces() const
{
    if(faces_) {
        return faces_->size();
    }
    return 0;
}

optix::GeometryTriangles GeometryTriangles::geometry() const
{
    return geometry_;
}

GeometryTriangles::operator optix::GeometryTriangles() const
{
    return geometry_;
}

optix::GeometryTriangles GeometryTriangles::operator->() const
{
    return geometry_;
}

}; //namespace optix_helpers

