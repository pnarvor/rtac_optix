#include <optix_helpers/Model.h>

namespace optix_helpers
{

ModelObj::ModelObj(const Context& context) :
    geomInstance_((*context)->createGeometryInstance())
{
}

void ModelObj::set_geometry(const Geometry& geometry)
{
    geometry_          = geometry;
    geometryTriangles_ = GeometryTriangles();
    geomInstance_->setGeometry(*geometry_);
}

void ModelObj::set_geometry(const GeometryTriangles& geometry)
{
    geometry_          = Geometry();
    geometryTriangles_ = geometry;
    geomInstance_->setGeometryTriangles(*geometryTriangles_);
    
    if(geometryTriangles_->points()) {
        geomInstance_["vertex_buffer"]->set(geometryTriangles_->points());
    }
    if(geometryTriangles_->faces()) {
        geomInstance_["index_buffer"]->set(geometryTriangles_->faces());
    }
    if(geometryTriangles_->normals()) {
        geomInstance_["normal_buffer"]->set(geometryTriangles_->normals());
    }
    if(geometryTriangles_->texture_coordinates()) {
        geomInstance_["texcoord_buffer"]->set(geometryTriangles_->texture_coordinates());
    }
}

void ModelObj::add_material(const Material& material)
{
    geomInstance_->setMaterialCount(materials_.size() + 1);
    geomInstance_->setMaterial(materials_.size(), *material);
    materials_.push_back(material);
}

optix::GeometryInstance ModelObj::geometry_instance() const
{
    return geomInstance_;
}

ModelObj::operator optix::GeometryInstance() const
{
    return geomInstance_;
}

optix::GeometryInstance ModelObj::operator->() const
{
    return geomInstance_;
}

};
