#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>

#include "QRSolver.hpp"
#include "OpenclKernels.hpp"

template <unsigned int block_size>
QRSolver<block_size>::QRSolver(const std::vector<int> &rowPointers_,
                               const std::vector<int> &colIndices_,
                               const std::vector<double> &nnzValues_):
    rowPointers(rowPointers_), colIndices(colIndices_), nnzValues(nnzValues_)
{
    nbrows = rowPointers.size() - 1;
    nbcols = *std::max_element(colIndices.begin(), colIndices.end()) + 1;
}

template <unsigned int block_size>
void QRSolver<block_size>::setOpencl(std::shared_ptr<cl::Context>& context_, std::shared_ptr<cl::CommandQueue>& queue_)
{
    context = context_;
    queue = queue_;
}

template <unsigned int block_size>
void QRSolver<block_size>::writeDataGPU()
{
    unsigned int bs = block_size;

    std::call_once(ocl_init, [&]() {
        d_rowPointers = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * rowPointers.size());
        d_colIndices = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(int) * colIndices.size());
        d_nnzValues = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nnzValues.size());
        d_rvMat = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nbrows * nbcols * bs * bs);
    });

    events.resize(4);
    err = queue->enqueueWriteBuffer(d_rowPointers, CL_FALSE, 0, rowPointers.size() * sizeof(int), rowPointers.data(), nullptr, &events[0]);
    err |= queue->enqueueWriteBuffer(d_colIndices, CL_FALSE, 0, colIndices.size() * sizeof(int), colIndices.data(), nullptr, &events[1]);
    err |= queue->enqueueWriteBuffer(d_nnzValues, CL_FALSE, 0, nnzValues.size() * sizeof(double), nnzValues.data(), nullptr, &events[2]);
    err |= queue->enqueueFillBuffer(d_rvMat, 0, 0, sizeof(double) * nbrows * nbcols * bs * bs, nullptr, &events[3]);
    cl::WaitForEvents(events);
}

template <unsigned int block_size>
void QRSolver<block_size>::decompose()
{
    unsigned int bs = block_size;
    OpenclKernels::qr_decomposition(nbrows, nbcols, bs, d_rowPointers, d_colIndices, d_nnzValues, d_rvMat);

    std::vector<double> rvMat(nbrows * nbcols * bs * bs);
    queue->enqueueReadBuffer(d_rvMat, CL_TRUE, 0, nbrows * nbcols * bs * bs * sizeof(double), rvMat.data());

    for(int i = 0; i < nbrows * bs; i++){
        for(int j = 0; j < bs; j++){
            std::cout << rvMat[i * nbcols * bs + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <unsigned int block_size>
void QRSolver<block_size>::solve()
{
}

template class QRSolver<3>;
