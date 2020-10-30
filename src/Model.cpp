#include <optix_helpers/Model.h>

namespace optix_helpers
{

Model::Ptr Model::New(const Context::ConstPtr& context)
{
    return Ptr(new Model(context));
}

Model::Model(const Context::ConstPtr& context) :
    geomInstance_((*context)->createGeometryInstance())
{
}

void Model::set_geometry(const Geometry::Ptr& geometry)
{
    geometry_          = geometry;
    geometryTriangles_ = GeometryTriangles::Ptr(NULL);
    geomInstance_->setGeometry(*geometry_);
}

void Model::set_geometry(const GeometryTriangles::Ptr& geometry)
{
    geometry_          = Geometry::Ptr(NULL);
    geometryTriangles_ = geometry;
    geomInstance_->setGeometryTriangles(*geometryTriangles_);
    
    if(geometryTriangles_->points()) {
        geomInstance_["vertex_buffer"]->set(geometryTriangles_->points()->buffer());
    }
    if(geometryTriangles_->faces()) {
        geomInstance_["index_buffer"]->set(geometryTriangles_->faces()->buffer());
    }
    if(geometryTriangles_->normals()) {
        geomInstance_["normal_buffer"]->set(geometryTriangles_->normals()->buffer());
    }
    if(geometryTriangles_->texture_coordinates()) {
        geomInstance_["texcoord_buffer"]->set(geometryTriangles_
            ->texture_coordinates()->buffer());
    }
}

void Model::add_material(const Material::Ptr& material)
{
    geomInstance_->setMaterialCount(materials_.size() + 1);
    geomInstance_->setMaterial(materials_.size(), *material);
    materials_.push_back(material);
}

optix::GeometryInstance Model::geometry_instance() const
{
    return geomInstance_;
}

Model::operator optix::GeometryInstance() const
{
    return geomInstance_;
}

optix::GeometryInstance Model::operator->() const
{
    return geomInstance_;
}

};
