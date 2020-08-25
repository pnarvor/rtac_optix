#ifndef _DEF_OPTIX_HELPERS_NVRTC_H_
#define _DEF_OPTIX_HELPERS_NVRTC_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <nvrtc.h>

namespace optix_helpers {

class Nvrtc
{
    public :

    using StringList = std::list<std::string>;
    static const std::string defaultCompileOptions;

    protected:
    
    // These are for convenience
    StringList includeDirs_;
    StringList compileOptions_;

    std::vector<const char*> nvrtc_options() const;
    nvrtcProgram program_;
    std::string compilationLog_;

    public:
    
    Nvrtc();
    Nvrtc(const Nvrtc& other);
    ~Nvrtc();
    void load_default_include_dirs();
    void load_default_compile_options();
    
    void clear_include_dirs();
    void clear_compile_options();
    void clear_program();
    void clear_all();

    void add_include_dirs(const std::string& dirs);
    void add_include_dirs(const StringList& dirs);
    void add_compile_options(const std::string& options);
    void add_compile_options(const StringList& options);
    
    std::string compile_cufile(const std::string& path,
                               const std::string& programName = "default_program",
                               const StringList& additionalHeaders = StringList(),
                               const StringList& headerNames = StringList());
    std::string compile(const std::string& source,
                        const std::string& programName = "default_program",
                        const StringList& additionalHeaders = StringList(),
                        const StringList& headerNames = StringList());
    void update_log();

    // getters
    StringList include_dirs()    const;
    StringList compile_options() const;
    std::string get_ptx()        const;
    nvrtcProgram release_program(); //This won't destroy the program. This is to
                                    //give the ownership to another scope.

    // static functions
    static StringList parse_option_string(const std::string& options,
                                          const std::string& separator);
    static std::string load_source_file(const std::string& path);
    static void check_error(nvrtcResult errorCode);
};

// For compatibility after class renaming
using NVRTC_Helper = Nvrtc;

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Nvrtc& nvrtc);

#endif //_DEF_OPTIX_HELPERS_NVRTC_H_
