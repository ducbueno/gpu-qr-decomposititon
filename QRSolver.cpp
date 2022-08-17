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
        d_b = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nbrows * bs * bs);
        d_x = cl::Buffer(*context, CL_MEM_READ_WRITE, sizeof(double) * nbcols * bs * bs);
    });

    events.resize(5);
    err = queue->enqueueWriteBuffer(d_rowPointers, CL_FALSE, 0, rowPointers.size() * sizeof(int), rowPointers.data(), nullptr, &events[0]);
    err |= queue->enqueueWriteBuffer(d_colIndices, CL_FALSE, 0, colIndices.size() * sizeof(int), colIndices.data(), nullptr, &events[1]);
    err |= queue->enqueueWriteBuffer(d_nnzValues, CL_FALSE, 0, nnzValues.size() * sizeof(double), nnzValues.data(), nullptr, &events[2]);
    err |= queue->enqueueFillBuffer(d_rvMat, 0, 0, sizeof(double) * nbrows * nbcols * bs * bs, nullptr, &events[3]);
    err |= queue->enqueueFillBuffer(d_b, 0, 0, sizeof(double) * nbrows * bs * bs, nullptr, &events[4]);
    cl::WaitForEvents(events);
}

template <unsigned int block_size>
void QRSolver<block_size>::decompose()
{
    unsigned int bs = block_size;
    int eye_idx = 0;

    for(int tile = 0; tile < nbcols; tile++){
        OpenclKernels::qr_decomposition(nbrows, nbcols, tile, bs, eye_idx, d_rowPointers, d_colIndices, d_nnzValues, d_rvMat, d_b);
    }

    // std::vector<double> rvMat(nbrows * nbcols * bs * bs);
    // queue->enqueueReadBuffer(d_rvMat, CL_TRUE, 0, nbrows * nbcols * bs * bs * sizeof(double), rvMat.data());

    // std::ofstream file;
    // file.open("../../data/rvMat.txt");

    // for(int i = 0; i < nbrows * bs; i++){
    //     for(int j = 0; j < nbcols * bs; j++){
    //         file << rvMat[i * nbcols * bs + j] << " ";
    //     }
    //     file << std::endl;
    // }

    // file.close();

    std::vector<double> x(nbcols * bs * bs);
    queue->enqueueReadBuffer(d_b, CL_TRUE, 0, nbcols * bs * bs * sizeof(double), x.data());

    for(int i = 0; i < nbcols * bs; i++){
        for(int j = 0; j < bs; j++){
            std::cout << x[i * bs + j] << " ";
        }
        std::cout << std::endl;
    }
}

template <unsigned int block_size>
void QRSolver<block_size>::solve()
{
    queue->enqueueCopyBuffer(d_b, d_x, 0, 0, sizeof(double) * nbcols * bs * bs);
}

template class QRSolver<3>;
