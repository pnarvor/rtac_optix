#include <optix_helpers/Source.h>

namespace optix_helpers {

Source::Source(const std::string& source, const std::string& name) :
    source_(source),
    name_(name)
{
}

Source::Source(const Source& other) :
    source_(other.source()),
    name_(other.name())
{
}

std::string Source::source() const
{
    return source_;
}

std::string Source::name() const
{
    return name_;
}

const char* Source::source_str() const
{
    return source_.c_str();
}

const char* Source::name_str() const
{
    return name_.c_str();
}

int Source::num_lines() const
{
    return std::count(source_.begin(), source_.end(), '\n');
}

std::ostream& Source::print(std::ostream& os) const
{
}

}; //namespace optix_helpers


std::ostream& operator<<(std::ostream& os, const optix_helpers::Source& source)
{
    os << "Optix source file " << source.name() << "\n";
    int Nlines = source.num_lines();
    int padWidth = std::to_string(Nlines - 1).size();
    std::istringstream iss(source.source());
    int lineIdx = 1;
    for(std::string line; std::getline(iss, line); lineIdx++) {
        os << std::setw(padWidth) << lineIdx << ' ' << line << '\n';
    }
    return os;
}
