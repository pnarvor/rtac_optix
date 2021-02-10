#include <rtac_optix/ProgramGroup.h>

namespace rtac { namespace optix {

OptixProgramGroupOptions ProgramGroup::default_options()
{
    return zero<OptixProgramGroupOptions>();
}

ProgramGroup::ProgramGroup(const Context::ConstPtr&        context,
                           const OptixProgramGroupDesc&    description,
                           const OptixProgramGroupOptions& options) :
    context_(context),
    program_(nullptr),
    description_(description),
    options_(options)
{
    this->store_entry_function_names();
}

void ProgramGroup::store_entry_function_names()
{
    // // This makes a copy of the entry function names in owned std::string.
    switch(description_.kind) {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            this->store_entry_function_name(entryFunctionNames_[0],
                                            &description_.raygen.entryFunctionName);
            break;
        case OPTIX_PROGRAM_GROUP_KIND_MISS:
            this->store_entry_function_name(entryFunctionNames_[0],
                                            &description_.miss.entryFunctionName);
            break;
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            this->store_entry_function_name(entryFunctionNames_[0],
                                            &description_.exception.entryFunctionName);
            break;
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            this->store_entry_function_name(entryFunctionNames_[0],
                                            &description_.callables.entryFunctionNameDC);
            this->store_entry_function_name(entryFunctionNames_[1],
                                            &description_.callables.entryFunctionNameCC);
            break;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            this->store_entry_function_name(entryFunctionNames_[0],
                                            &description_.hitgroup.entryFunctionNameCH);
            this->store_entry_function_name(entryFunctionNames_[1],
                                            &description_.hitgroup.entryFunctionNameAH);
            this->store_entry_function_name(entryFunctionNames_[2],
                                            &description_.hitgroup.entryFunctionNameIS);
            break;
        default:
            std::ostringstream oss;
            oss << "Invalid program group kind : " << description_.kind;
            throw std::runtime_error(oss.str());
            break;
    }
}

void ProgramGroup::store_entry_function_name(std::string& dst, const char** src)
{
    if(!src || !*src) return;
    dst = std::string(*src);
    *src = dst.c_str();
}

ProgramGroup::~ProgramGroup()
{
    try {
        if(program_ != nullptr)
            OPTIX_CHECK( optixProgramGroupDestroy(program_) );
    }
    catch(const std::runtime_error& e) {
        std::cerr << "Caught exception during rtac::optix::ProgramGroup destruction : " 
                  << e.what() << std::endl;
    }
}

ProgramGroup::Ptr ProgramGroup::Create(const Context::ConstPtr&        context,
                                       const OptixProgramGroupDesc&    description,
                                       const OptixProgramGroupOptions& options)
{
    return Ptr(new ProgramGroup(context, description, options));
}

OptixProgramGroup ProgramGroup::build()
{
    if(program_ != nullptr)
        return program_;

    OPTIX_CHECK( optixProgramGroupCreate(
        *context_, &description_, 1, &options_,
        nullptr, nullptr, // These are logging related, log will also
                          // be written in context log, but with less
                          // tracking information (TODO Fix this).
        &program_));
    return program_;
}

ProgramGroup::operator OptixProgramGroup()
{
    this->build();
    return program_;
}

}; //namespace optix {
}; //namespace rtac { namespace optix {
