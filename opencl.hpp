#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_TARGET_OPENCL_VERSION 120   // indicate OpenCL 1.2 is used
#define CL_HPP_TARGET_OPENCL_VERSION 120 // indicate OpenCL 1.2 is used
#define CL_HPP_MINIMUM_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#include <CL/opencl.hpp>
#include <string>

std::string getErrorString(cl_int error);
