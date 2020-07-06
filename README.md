# NVIDIA OptiX library installation helper

This is a cmake script allowing to install an interface target pointing to the NVIDIA OptiX SDK.

The NVIDIA OptiX SDK is currently distributed without any installation
procedure, and therefore cannot be properly added into a cmake-based workflow.

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


