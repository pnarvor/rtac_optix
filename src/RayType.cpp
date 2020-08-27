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

RayType::RayType() :
    Handle<RayTypeObj>()
{}

RayType::RayType(Index rayTypeIndex, const Source& definition) :
    Handle<RayTypeObj>(new RayTypeObj(rayTypeIndex, definition))
{}

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType)
{
    if(!rayType) {
        os << "Empty ray.\n";
        return os;
    }
    os << "RayType " << (int)rayType->index() << "\n" << rayType->definition() << "\n";
    return os;
}



