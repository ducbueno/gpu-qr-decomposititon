#include <vector>
#include <fstream>
#include <iostream>
#include <memory>

#include "OpenclKernels.hpp"
#include "QRSolver.hpp"

template<typename T>
void read_vec(const std::string &fname, std::vector<T> &temp){
    T value;
    std::ifstream input(fname.c_str());

    while(input >> value){
        temp.push_back(value);
    }
    input.close();
}

void initOpenCL(std::shared_ptr<cl::Context> &context, std::shared_ptr<cl::CommandQueue> &queue){
    int platformID = 0;
    int deviceID = 0;
    cl_int err = CL_SUCCESS;

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    cl::Platform::get(&platforms);
    platforms[platformID].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    context = std::make_shared<cl::Context>(devices[deviceID]);
    queue.reset(new cl::CommandQueue(*context, devices[deviceID], 0, &err));

    OpenclKernels::init(context.get(), queue.get(), devices);
}

int main(int argc, char** argv){
    std::vector<int> rowPointers, colIndices;
    std::vector<double> nnzValues;

    std::shared_ptr<cl::Context> context;
    std::shared_ptr<cl::CommandQueue> queue;
    initOpenCL(context, queue);

    read_vec<int>("/hdd/mysrc/qr_decomposition/data/spe1case1_submat133_colIndices.txt", colIndices);
    read_vec<int>("/hdd/mysrc/qr_decomposition/data/spe1case1_submat133_rowPointers.txt", rowPointers);
    read_vec<double>("/hdd/mysrc/qr_decomposition/data/spe1case1_submat133_nnzValues.txt", nnzValues);

    QRSolver<3> solver(rowPointers, colIndices, nnzValues);
    solver.setOpencl(context, queue);
    solver.writeDataGPU();

    return 0;
}
