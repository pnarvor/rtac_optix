#include <optix_helpers/RayType.h>

namespace optix_helpers {

RayType::RayType(Index rayTypeIndex, const Source& definition) :
    rayTypeIndex_(rayTypeIndex),
    definition_(definition)
{
}

RayType::RayType(const RayType& other) :
    rayTypeIndex_(other.index()),
    definition_(other.definition())
{
}

RayType::Index RayType::index() const
{
    return rayTypeIndex_;
}

Source RayType::definition() const
{
    return definition_;
}

bool RayType::is_defined() const
{
    return rayTypeIndex_ >= 0;
}


}; //namespace optix_helpers

