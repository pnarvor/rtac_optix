#include <optix_helpers/ProgramManager.h>

namespace optix_helpers {

StringPtr StringPtrCreate(const std::string& other)
{
    return StringPtr(new std::string(other));
}

Program::Program() :
    cuString_(NULL),
    ptxString_(NULL),
    name_("None"),
    program_(0)
{
}

Program::Program(const StringPtr& cuString, const StringPtr& ptxString,
                 const std::string& functionName, const optix::Program& program) :
    cuString_(cuString),
    ptxString_(ptxString),
    name_(functionName),
    program_(program)
{
}

Program::Program(const Program& other) :
    cuString_(other.cuString_),
    ptxString_(other.ptxString_),
    name_(other.name_),
    program_(other.program_)
{
}

StringPtr Program::cu_string() const
{
    return cuString_;
}

StringPtr Program::ptx_string() const
{
    return ptxString_;
}

std::string Program::name() const
{
    return name_;
}

optix::Program Program::program() const
{
    return program_;
}

bool Program::operator!() const
{
    return !this->program_;
}

//static functions
void Program::print_source(const std::string& source, std::ostream& os)
{
    int Nlines = std::count(source.begin(), source.end(), '\n');
    int padWidth = std::to_string(Nlines - 1).size();
    std::istringstream iss(source);
    int lineIdx = 1;
    for(std::string line; std::getline(iss, line); lineIdx++) {
        os << std::setw(padWidth) << lineIdx << ' ' << line << '\n';
    }
}

std::string Program::print_source(const std::string& source)
{
    std::ostringstream oss;
    Program::print_source(source, oss);
    return oss.str();
}

//ProgramManager definition //////////////////////////////
ProgramManager::ProgramManager(const optix::Context& context) :
    context_(context)
{
}

ProgramManager::ProgramManager(const ProgramManager& other) :
    context_(other.context_),
    nvrtc_(other.nvrtc_)
{
}

Program ProgramManager::from_cufile(const std::string& path,
                                    const std::string& functionName)
{
    return this->from_custring(Nvrtc::load_source_file(path), functionName);
}

Program ProgramManager::from_custring(const std::string& cuString,
                                      const std::string& functionName)
{
    try {
        auto ptx = nvrtc_.compile(cuString, functionName);
        optix::Program program = context_->createProgramFromPTXString(ptx, functionName);
        return Program(StringPtrCreate(cuString),
                       StringPtrCreate(ptx),
                       functionName, program);
    }
    catch(const std::runtime_error& e) {
        std::ostringstream os;
        Program::print_source(cuString, os);
        os << "\n" << e.what();
        throw std::runtime_error(os.str());
    }
}

} //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program)
{
    optix_helpers::StringPtr source(program.cu_string());
    if(!source) {
        source = program.ptx_string();
        if(!source) {
            os << "Invalid or Uninitialized program\n";
            return os;
        }
    }
    os << "Optix program (function name : " << program.name() << ")\n";
    optix_helpers::Program::print_source(*source, os);
    return os;
}


