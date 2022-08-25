#ifndef OPENCL_KERNELS_HPP
#define OPENCL_KERNELS_HPP

#include <CL/opencl.hpp>
#include <string>
#include <memory>
#include "opencl.hpp"

using qr_decomposition_kernel_type = cl::KernelFunctor<const int, const int, const int, const int, const int,
                                                       cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::Buffer&, cl::LocalSpaceArg>;
using solve_kernel_type = cl::KernelFunctor<const int, const int, cl::Buffer&, cl::Buffer&, cl::Buffer&>;

class OpenclKernels
{
private:
    static int verbosity;
    static cl::CommandQueue *queue;
    static bool initialized;

    static std::unique_ptr<qr_decomposition_kernel_type> qr_decomposition_k;
    static std::unique_ptr<solve_kernel_type> solve_k;

    OpenclKernels(){}; // diasable instantiation

public:
    static const std::string qr_decomposition_str;
    static const std::string solve_str;

    static void init(cl::Context *context, cl::CommandQueue *queue, std::vector<cl::Device>& devices);
    static void qr_decomposition(int nbrows, int nbcols, int tile, int block_size, int eye_idx,
                                 cl::Buffer& sh_ptrs, cl::Buffer& sh_inds, cl::Buffer& sh_vals, cl::Buffer& rv_mat, cl::Buffer& x);
    static void solve(int nbcols, int block_size, cl::Buffer& rv_mat, cl::Buffer& b, cl::Buffer& x);
};

#endif
