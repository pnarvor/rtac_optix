#ifndef _DEF_OPTIX_HELPERS_NVRTC_H_
#define _DEF_OPTIX_HELPERS_NVRTC_H_

#include <iostream>
#include <vector>
#include <list>

namespace optix {

class NVRTC_Helper
{
    public :

    using StringList = std::list<std::string>;
    static const std::string defaultCompileOptions;

    protected:
    
    StringList includeDirs_;
    StringList compileOptions_;

    public:
    
    NVRTC_Helper();
    void load_default_include_dirs();
    void load_default_compile_options();
    
    void clear_include_dirs();
    void clear_compile_options();
    void clear_all();

    void add_include_dirs(const std::string& dirs);
    void add_include_dirs(StringList&& dirs);
    void add_compile_options(const std::string& options);
    void add_compile_options(StringList&& options);
    
    // simple getters
    StringList include_dirs()    const;
    StringList compile_options() const;

    // static functions
    static StringList parse_option_string(const std::string& options,
                                          const std::string& separator);
};

}; //namespace optix

std::ostream& operator<<(std::ostream& os, const optix::NVRTC_Helper& nvrtc);

#endif //_DEF_OPTIX_HELPERS_NVRTC_H_
