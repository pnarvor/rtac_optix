#include <optix_helpers/NVRTC_Helper.h>
#include <list>

#ifndef NVRTC_INCLUDE_DIRS
#define NVRTC_INCLUDE_DIRS ""
#endif

namespace optix {

NVRTC_Helper::NVRTC_Helper()
{
    this->load_default_include_dirs();
    this->load_default_compile_options();
}

void NVRTC_Helper::load_default_include_dirs()
{
    // NVRTC_INCLUDE_DIRS was defined at optix installation and contains mostly
    // some paths to th optix SDK.
    includeDirs_.splice(includeDirs_.end(),
        parse_option_string(NVRTC_INCLUDE_DIRS, ";"));
}

const std::string NVRTC_Helper::defaultCompileOptions(
    "-arch=compute_30 -use_fast_math -lineinfo -default-device -rdc=true -D__x86_64");
void NVRTC_Helper::load_default_compile_options()
{
    this->add_compile_options(defaultCompileOptions);
}

void NVRTC_Helper::clear_include_dirs()
{
    includeDirs_.clear();
}

void NVRTC_Helper::clear_compile_options()
{
    compileOptions_.clear();
}

void NVRTC_Helper::clear_all()
{
    this->clear_include_dirs();
    this->clear_compile_options();
}

void NVRTC_Helper::add_include_dirs(const std::string& dirs)
{
    this->add_include_dirs(parse_option_string(dirs, " "));
}

void NVRTC_Helper::add_include_dirs(StringList&& dirs)
{
    includeDirs_.splice(includeDirs_.end(), dirs);
}

void NVRTC_Helper::add_compile_options(const std::string& options)
{
    this->add_compile_options(parse_option_string(options, " "));
}

void NVRTC_Helper::add_compile_options(StringList&& options)
{
    compileOptions_.splice(compileOptions_.end(), options);
}

NVRTC_Helper::StringList NVRTC_Helper::include_dirs() const
{
    return includeDirs_;
}

NVRTC_Helper::StringList NVRTC_Helper::compile_options() const
{
    return compileOptions_;
}

NVRTC_Helper::StringList NVRTC_Helper::parse_option_string(const std::string& options,
                                                           const std::string& separator)
{
    StringList parsedOptions;

    // removing unwanted quotes (results from setting the definition through
    // cmake. See if better way)
    std::string opts(options);
    if(opts[0] == '\"')
        opts.erase(0,1);
    if(opts[opts.size()-1] == '\"')
        opts.pop_back();

    parsedOptions.clear();
    size_t pos;
    while((pos = opts.find(separator)) != std::string::npos) {
        parsedOptions.push_back(opts.substr(0, pos));
        opts.erase(0, pos + separator.size());
    }
    if(opts.size() > 0)
        parsedOptions.push_back(opts);

    return parsedOptions;
}

}; //namespace optix

std::ostream& operator<<(std::ostream& os, const optix::NVRTC_Helper& nvrtc)
{
    os << "NVRTC compile options :\n";
    os << "Include dirs :";
    for(auto dir : nvrtc.include_dirs()) {
        os << "\n -I" << dir;
    }
    os << "\nCompile options :";
    for(auto dir : nvrtc.compile_options()) {
        os << "\n " << dir;
    }
    return os;
}

