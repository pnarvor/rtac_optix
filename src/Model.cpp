#include <optix_helpers/Model.h>

namespace optix_helpers
{

ModelObj::ModelObj(const optix::GeometryInstance& geomInstance) :
    geomInstance_(geomInstance)
{
}

void ModelObj::set_geometry(const Geometry& geometry)
{
    geometry_          = geometry;
    geometryTriangles_ = GeometryTriangles();
    geomInstance_->setGeometry(geometry_);
}

void ModelObj::set_geometry(const GeometryTriangles& geometry)
{
    geometry_          = Geometry();
    geometryTriangles_ = geometry;
    geomInstance_->setGeometryTriangles(geometryTriangles_);
}

void ModelObj::add_material(const Material& material)
{
    geomInstance_->setMaterialCount(materials_.size() + 1);
    geomInstance_->setMaterial(materials_.size(), material);
    materials_.push_back(material);
}

optix::GeometryInstance ModelObj::geometry_instance() const
{
    return geomInstance_;
}

Model::Model() :
    Handle<ModelObj>()
{}

Model::Model(const optix::GeometryInstance& geomInstance) :
    Handle<ModelObj>(new ModelObj(geomInstance))
{}


};
