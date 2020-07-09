#include <optix_helpers/NVRTC_Helper.h>

#ifndef NVRTC_INCLUDE_DIRS
#define NVRTC_INCLUDE_DIRS ""
#endif

namespace optix_helpers {

NVRTC_Helper::NVRTC_Helper() :
    program_(0)
{
    this->load_default_include_dirs();
    this->load_default_compile_options();
}

NVRTC_Helper::~NVRTC_Helper()
{
}

void NVRTC_Helper::load_default_include_dirs()
{
    // NVRTC_INCLUDE_DIRS was defined at optix installation and contains mostly
    // some paths to th optix SDK.
    this->add_include_dirs(parse_option_string(NVRTC_INCLUDE_DIRS, ";"));
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

void NVRTC_Helper::clear_program()
{
    if(program_ != 0) {
        check_error(nvrtcDestroyProgram(&program_));
        program_ = 0;
    }
}

void NVRTC_Helper::clear_all()
{
    this->clear_include_dirs();
    this->clear_compile_options();
    this->clear_program();
}

void NVRTC_Helper::add_include_dirs(const std::string& dirs)
{
    this->add_include_dirs(parse_option_string(dirs, " "));
}

void NVRTC_Helper::add_include_dirs(const StringList& dirs)
{
    for(auto dir : dirs) {
        compileOptions_.push_back("-I" + dir);
    }
}

void NVRTC_Helper::add_compile_options(const std::string& options)
{
    this->add_compile_options(parse_option_string(options, " "));
}

void NVRTC_Helper::add_compile_options(const StringList& options)
{
    for(auto opt : options) {
        compileOptions_.push_back(opt);
    }
}

std::string NVRTC_Helper::compile(const std::string& source,
                                  const char* programName)
{
    std::vector<const char*> options = this->nvrtc_options();
    
    this->clear_program();
    check_error(nvrtcCreateProgram(&program_, source.c_str(), programName,
                                   0, NULL, NULL));
    try {
        check_error(nvrtcCompileProgram(program_, options.size(), options.data()));
    }
    catch(const std::runtime_error& e) {
        this->update_log();
        throw std::runtime_error("NVRTC compilation failed :\n" + compilationLog_);
    }
    this->update_log();

    return get_ptx();
}

void NVRTC_Helper::update_log()
{
    if(program_ == 0)
        return;
    size_t logSize = 0;
    check_error(nvrtcGetProgramLogSize(program_, &logSize));
    if(logSize > 1) {
        compilationLog_.resize(logSize);
        check_error(nvrtcGetProgramLog(program_, &compilationLog_[0]));
    }
}

NVRTC_Helper::StringList NVRTC_Helper::include_dirs() const
{
    return includeDirs_;
}

NVRTC_Helper::StringList NVRTC_Helper::compile_options() const
{
    return compileOptions_;
}

std::string NVRTC_Helper::get_ptx() const
{
    std::string ptx("");

    if(program_ == 0)
        return ptx;

    size_t ptxSize;
    check_error(nvrtcGetPTXSize(program_, &ptxSize));
    ptx.resize(ptxSize);
    check_error(nvrtcGetPTX(program_, &ptx[0]));

    return ptx;
}

std::vector<const char*> NVRTC_Helper::nvrtc_options() const
{
    // build an array of const char* because that's what expects NVRTC as input
    std::vector<const char*> options(includeDirs_.size() + compileOptions_.size());
    int idx = 0;
    for(auto& dir : includeDirs_) {
        options[idx] = dir.c_str();
        idx++;
    }
    for(auto& opt : compileOptions_) {
        options[idx] = opt.c_str();
        idx++;
    }
    return options;
}

// static methods
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

std::string NVRTC_Helper::load_source_file(const std::string& path)
{
    std::ifstream f;
    f.open(path, std::ios::in);
    if(!f)
        throw std::runtime_error("Cloud not open " + path);
    std::ostringstream contents;
    contents << f.rdbuf();
    f.close();

    return contents.str();
}

void NVRTC_Helper::check_error(nvrtcResult errorCode)
{
    if(errorCode != NVRTC_SUCCESS) {
        throw std::runtime_error("NVRTC compilation error : " +
                                 std::string(nvrtcGetErrorString(errorCode)));
    }
}

}; //namespace optix

std::ostream& operator<<(std::ostream& os, const optix_helpers::NVRTC_Helper& nvrtc)
{
    os << "NVRTC compile options :\n";
    os << "Include dirs :";
    for(auto dir : nvrtc.include_dirs()) {
        os << "\n " << dir;
    }
    os << "\nCompile options :";
    for(auto dir : nvrtc.compile_options()) {
        os << "\n " << dir;
    }
    return os;
}

