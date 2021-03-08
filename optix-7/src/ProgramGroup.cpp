#include <rtac_optix/ProgramGroup.h>

namespace rtac { namespace optix {

const char* ProgramGroup::Function::Raygen               = "raygen";
const char* ProgramGroup::Function::Miss                 = "miss";
const char* ProgramGroup::Function::Exception            = "exception";
const char* ProgramGroup::Function::Intersection         = "intersection";
const char* ProgramGroup::Function::AnyHit               = "anyhit";
const char* ProgramGroup::Function::ClosestHit           = "closesthit";
const char* ProgramGroup::Function::DirectCallable       = "direct_callable";
const char* ProgramGroup::Function::ContinuationCallable = "continuation_callable";

ProgramGroup::Description ProgramGroup::empty_description(const Kind& kind,  
                                                          unsigned int flags)
{
    auto description = zero<Description>();
    description.kind  = kind;
    description.flags = flags;
    return description;
}

ProgramGroup::Options ProgramGroup::default_options()
{
    return zero<Options>();
}

ProgramGroup::ProgramGroup(const Context::ConstPtr& context,
                           Kind kind, unsigned int flags,
                           const Options& options) :
    context_(context),
    description_(empty_description(kind, flags)),
    options_(options)
{}

ProgramGroup::Ptr ProgramGroup::Create(const Context::ConstPtr& context,
                                       Kind kind, unsigned int flags,
                                       const Options& options)
{
    return Ptr(new ProgramGroup(context, kind, flags, options));
}

ProgramGroup::~ProgramGroup()
{
    try {
        this->clean();
    }
    catch(const std::runtime_error& e) {
        std::cerr << "Caught exception during rtac::optix::ProgramGroup destruction : " 
                  << e.what() << std::endl;
    }
}

void ProgramGroup::update_description() const
{
    auto f = functions_.end();
    switch(this->kind()) {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            f = this->function(Function::Raygen);
            description_.raygen.module            = *f->second.module;
            description_.raygen.entryFunctionName = f->second.name.c_str();
            break;
        case OPTIX_PROGRAM_GROUP_KIND_MISS:
            f = this->function(Function::Miss);
            description_.miss.module            = *f->second.module;
            description_.miss.entryFunctionName = f->second.name.c_str();
            break;
        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            f = this->function(Function::Exception);
            description_.exception.module            = *f->second.module;
            description_.exception.entryFunctionName = f->second.name.c_str();
            break;
        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            try {
                f = this->function(Function::Intersection);
                description_.hitgroup.moduleIS            = *f->second.module;
                description_.hitgroup.entryFunctionNameIS = f->second.name.c_str();
            }
            catch(const FunctionNotFound&) {} // ignoring here but check f afterwards
            try {
                f = this->function(Function::AnyHit);
                description_.hitgroup.moduleAH            = *f->second.module;
                description_.hitgroup.entryFunctionNameAH = f->second.name.c_str();
            }
            catch(const FunctionNotFound&) {} // ignoring here but check f afterwards
            try {
                f = this->function(Function::ClosestHit);
                description_.hitgroup.moduleCH            = *f->second.module;
                description_.hitgroup.entryFunctionNameCH = f->second.name.c_str();
            }
            catch(const FunctionNotFound&) {} // ignoring here but check f afterwards
            if(f == functions_.end()) {
                // f not set, no function was successfully retrived through the
                // function method, so no program was set.
                std::ostringstream oss;
                oss << "ProgramGroup kind is HITGROUP but no "
                    << "intersection anyhit or closesthit function was set.";
                throw std::runtime_error(oss.str());
            }
            break;
        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            try {
                f = this->function(Function::DirectCallable);
                description_.callables.moduleDC            = *f->second.module;
                description_.callables.entryFunctionNameDC = f->second.name.c_str();
            }
            catch(const FunctionNotFound&) {} // ignoring here but check f afterwards
            try {
                f = this->function(Function::ContinuationCallable);
                description_.callables.moduleCC            = *f->second.module;
                description_.callables.entryFunctionNameCC = f->second.name.c_str();
            }
            catch(const FunctionNotFound&) {} // ignoring here but check f afterwards
            if(f == functions_.end()) {
                // f not set, no function was successfully retrived through the
                // function method, so no program was set.
                std::ostringstream oss;
                oss << "ProgramGroup kind is CALLABLES but no "
                    << "direct_callable or continuation_callable function was set.";
                throw std::runtime_error(oss.str());
            }
            break;
        default:
            throw std::runtime_error(
                "Unknown program group kind : check OptiX version");
            break;
    }
}

void ProgramGroup::do_build() const
{
    this->update_description();
    OPTIX_CHECK( optixProgramGroupCreate(
        *context_, &description_, 1, &options_,
        nullptr, nullptr, // These are logging related, log will also
                          // be written in context log, but with less
                          // tracking information (TODO Fix this).
        &optixObject_));
}

void ProgramGroup::clean() const
{
    if(optixObject_ != nullptr)
        OPTIX_CHECK( optixProgramGroupDestroy(optixObject_) );
}

const ProgramGroup::Description& ProgramGroup::description() const
{
    return description_;
}

const ProgramGroup::Options& ProgramGroup::options() const
{
    return options_;
}

ProgramGroup::Description& ProgramGroup::description()
{
    this->bump_version();
    return description_;
}

ProgramGroup::Options& ProgramGroup::options()
{
    this->bump_version();
    return options_;
}

ProgramGroup::Kind ProgramGroup::kind() const
{
    return description_.kind;
}

unsigned int ProgramGroup::flags() const
{
    return description_.flags;
}

void ProgramGroup::set_kind(Kind kind)
{
    if(this->kind() != kind) {
        // We do something only if the program kind changes.
        this->description() = empty_description(kind, this->flags());
        // If the kind changes, functions_ are unrelated to the new kind
        functions_.clear();
    }
}

void ProgramGroup::add_function(const std::string& kind, const Function& function)
{
    if(!function.module) {
        throw std::runtime_error("Function module is null");
    }
    this->bump_version();
    functions_[kind] = function;
    // This allows to rebuild this object if the module changes.
    this->add_dependency(function.module);
}

ProgramGroup::Functions::const_iterator ProgramGroup::function(const std::string& kind) const
{
    auto it = functions_.find(kind);
    if(it == functions_.end()) {
        throw FunctionNotFound(kind);
    }
    return it;
}

void ProgramGroup::set_raygen(const Function& function)
{
    this->set_kind(OPTIX_PROGRAM_GROUP_KIND_RAYGEN);
    this->add_function(Function::Raygen, function);
}

void ProgramGroup::set_miss(const Function& function)
{
    this->set_kind(OPTIX_PROGRAM_GROUP_KIND_MISS);
    this->add_function(Function::Miss, function);
}

void ProgramGroup::set_exception(const Function& function)
{
    this->set_kind(OPTIX_PROGRAM_GROUP_KIND_EXCEPTION);
    this->add_function(Function::Exception, function);
}

void ProgramGroup::set_intersection(const Function& function)
{
    this->set_kind(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
    this->add_function(Function::Intersection, function);
}

void ProgramGroup::set_anyhit(const Function& function)
{
    this->set_kind(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
    this->add_function(Function::AnyHit, function);
}

void ProgramGroup::set_closesthit(const Function& function)
{
    this->set_kind(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
    this->add_function(Function::ClosestHit, function);
}

void ProgramGroup::set_direct_callable(const Function& function)
{
    this->set_kind(OPTIX_PROGRAM_GROUP_KIND_CALLABLES);
    this->add_function(Function::DirectCallable, function);
}

void ProgramGroup::set_continuation_callable(const Function& function)
{
    this->set_kind(OPTIX_PROGRAM_GROUP_KIND_CALLABLES);
    this->add_function(Function::ContinuationCallable, function);
}

}; //namespace optix {
}; //namespace rtac { namespace optix {
