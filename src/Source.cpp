#include <optix_helpers/Source.h>

namespace optix_helpers {

SourceObj::SourceObj(const std::string& source, const std::string& name) :
    source_(source),
    name_(name)
{
}

std::string SourceObj::source() const
{
    return source_;
}

std::string SourceObj::name() const
{
    return name_;
}

const char* SourceObj::source_str() const
{
    return source_.c_str();
}

const char* SourceObj::name_str() const
{
    return name_.c_str();
}

int SourceObj::num_lines() const
{
    return std::count(source_.begin(), source_.end(), '\n');
}

Source::Source() :
    Handle<SourceObj>()
{}

Source::Source(const std::string& source, const std::string& name) :
    Handle<SourceObj>(new SourceObj(source, name))
{}

}; //namespace optix_helpers

optix_helpers::Sources operator+(const optix_helpers::Sources& lhs, 
                                 const optix_helpers::Sources& rhs)
{
    optix_helpers::Sources res = lhs;
    res.insert(res.end(), rhs.begin(), rhs.end());
    return res;
}

std::ostream& operator<<(std::ostream& os, const optix_helpers::Source& source)
{
    os << "Optix source file " << source->name() << "\n";
    int Nlines = source->num_lines();
    int padWidth = std::to_string(Nlines - 1).size();
    std::istringstream iss(source->source());
    int lineIdx = 1;
    for(std::string line; std::getline(iss, line); lineIdx++) {
        os << std::setw(padWidth) << lineIdx << ' ' << line << '\n';
    }
    return os;
}
