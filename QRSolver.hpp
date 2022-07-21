#ifndef SPAI_HPP
#define SPAI_HPP

#include <memory>
#include <mutex>

#include "opencl.hpp"

template <unsigned int block_size>
class QRSolver
{
private:
    unsigned int bs;
    int nbrows, nbcols;
    std::vector<int> rowPointers, colIndices;
    std::vector<double> nnzValues;

    std::once_flag ocl_init;

    cl_int err;
    std::vector<cl::Event> events;
    std::shared_ptr<cl::Context> context;
    std::shared_ptr<cl::CommandQueue> queue;
    cl::Buffer d_rowPointers, d_colIndices;
    cl::Buffer d_nnzValues, d_rvMat;

public:
    QRSolver(const std::vector<int> &rowPointers_,
             const std::vector<int> &colIndices_,
             const std::vector<double> &nnzValues_);

    void setOpencl(std::shared_ptr<cl::Context>& context_, std::shared_ptr<cl::CommandQueue>& queue_);
    void writeDataGPU();
    void decompose();
    void solve();
};

#endif
