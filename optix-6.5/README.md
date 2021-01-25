# NVIDIA OptiX helpers

Helpers for using the OptiX SDK from NVIDIA, such as :
    - CMake wrapper : the OptiX SDK is shipped without a proper installation
      procedure. This makes the use of the OptiX library in external projects
      tedious at best (+ it is not sure if we have the right to distribute the
      SDK with our code, so better make a proper interface). Installation of
      this package will create cmake target you can import in your own project.
    - Various helpers to help with the use of the SDK (cuda jit compiler with
      default options, etc...)

# Usage :

## Install OptiX (this part of the guide is for linux only)

First download the OptiX SDK from NVIDIA website
(https://developer.nvidia.com/optix), and "install" it using using the
downloaded bash script.

/!\ Make sure to get a OptiX SDK compatible with your NVIDIA driver version
(you can check the driver version using the nvidia-smi command).


## Now make a real installation using this module

mkdir build && cd build
cmake -DOPTIX_PATH=<path to your optix SDK> -DCMAKE_INSTALL_PREFIX=<path where you want cmake to install the optix interface (leave empty if you do not know)> ..
make install


