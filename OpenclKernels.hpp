#ifndef OPENCL_KERNELS_HPP
#define OPENCL_KERNELS_HPP

#include <CL/opencl.hpp>
#include <string>
#include <memory>
#include "opencl.hpp"

using qr_decomposition_kernel_type = cl::KernelFunctor<const int, const int, const int, const int, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg>;

class OpenclKernels
{
private:
    static int verbosity;
    static cl::CommandQueue *queue;
    static bool initialized;

    static std::unique_ptr<qr_decomposition_kernel_type> qr_decomposition_k;

    OpenclKernels(){}; // diasable instantiation

public:
    static const std::string qr_decomposition_str;

    static void init(cl::Context *context, cl::CommandQueue *queue, std::vector<cl::Device>& devices);
    static void qr_decomposition(int nbrows, int nbcols, int tile, int block_size, cl::Buffer& sh_ptrs, cl::Buffer& sh_inds, cl::Buffer& sh_vals, cl::Buffer& rv_mat);
};

#endif
