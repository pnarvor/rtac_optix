#include <optix_helpers/RayType.h>

namespace optix_helpers {

RayType::RayType(Index rayTypeIndex, const Source::ConstPtr& definition) :
    rayTypeIndex_(rayTypeIndex),
    definition_(definition)
{}

RayType::Index RayType::index() const
{
    return rayTypeIndex_;
}

RayType::operator RayType::Index() const
{
    return this->index();
}

Source::ConstPtr RayType::definition() const
{
    return definition_;
}

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType)
{
    if(rayType != optix_helpers::RayType::uninitialized)
        os << "RayType " << (int)rayType.index() << "\n" << rayType.definition() << "\n";
    else
        os << "Empty RayType.\n";

    return os;
}



