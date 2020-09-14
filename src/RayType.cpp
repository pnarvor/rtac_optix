#include <optix_helpers/RayType.h>

namespace optix_helpers {

RayTypeObj::RayTypeObj(Index rayTypeIndex, const Source& definition) :
    rayTypeIndex_(rayTypeIndex),
    definition_(definition)
{}

Source RayTypeObj::definition() const
{
    return definition_;
}

RayTypeObj::Index RayTypeObj::index() const
{
    return rayTypeIndex_;
}

RayTypeObj::operator Index() const
{
    return this->index();
}

RayType::RayType() :
    Handle<RayTypeObj>()
{}

RayType::operator Index() const
{
    return (*this)->index();
}

RayType::RayType(Index rayTypeIndex, const Source& definition) :
    Handle<RayTypeObj>(new RayTypeObj(rayTypeIndex, definition))
{}

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType)
{
    if(rayType)
        os << "RayType " << (int)rayType->index() << "\n" << rayType->definition() << "\n";
    else
        os << "Empty RayType.\n";

    return os;
}



