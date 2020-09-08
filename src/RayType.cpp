#include <optix_helpers/RayType.h>

namespace optix_helpers {

RayTypeObj::RayTypeObj(Index rayTypeIndex, const Source& definition) :
    rayTypeIndex_(rayTypeIndex),
    definition_(definition)
{}

RayTypeObj::Index RayTypeObj::index() const
{
    return rayTypeIndex_;
}

Source RayTypeObj::definition() const
{
    return definition_;
}

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType)
{
    os << "RayType " << (int)rayType->index() << "\n" << rayType->definition() << "\n";
    return os;
}



