//Use cl::vector instead of STL version
#define __NO_STD_VECTOR
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <windows.h>

#include "../inc/convolution.h"


#include <oclutils.hpp>

void read_input(const std::string& data_file, parameters& p, std::vector<float>& IPs, std::vector<float>& Fs, std::vector<float>& Bs, std::vector<float>& OPs);
void read_reference_output(const std::string& ref_fname, std::vector<float>& OPs);
float validate_result(const std::vector<float>& refOPs, const std::vector<float>& OPs);

unsigned round_up_div(unsigned a, unsigned b) {
    return static_cast<int>(ceil((double)a / b));
}

int main() {
    using namespace std::chrono;
    using clock = steady_clock;

    std::vector<float> IPs, Fs, Bs, OPs, refOPs;

    parameters p = { 0 };

    static const float error_threshold = 0.0f;

    read_input("dat/input.dat", p, IPs, Fs, Bs, OPs);
    read_reference_output("dat/reference_output.dat", refOPs);

    // Create the convolution
    convolution* c = convolution_new(&p);

    // The benchmark parameters
    unsigned int N_iter = 1;

    clock::duration total_time = clock::duration::zero();
    for (unsigned int i = 0; i < N_iter; i++) {
        auto start_time = clock::now();

        // run the convolution
        //convolution_run(c);
        fast_convolution_run(c);

        total_time += (clock::now() - start_time);
        if (i % 50 == 0) {
            // update the result in p.OPs
            convolution_result(c);
            float total_error = validate_result(refOPs, OPs);
            std::cout << "Validated: " << std::boolalpha << (total_error <= error_threshold) << std::endl;
        }
    }
    std::cout << "Running time: " << duration_cast<milliseconds>(total_time).count() / (double)N_iter << " ms" << std::endl;

    // destroy the convolution
    convolution_destroy(c);


    try {
#pragma region Initialize GPU

        cl::Context context;
        if (!oclCreateContextBy(context, "nvidia")) {
            throw cl::Error(CL_INVALID_CONTEXT, "Failed to create a valid context!");
        }

        // Query devices from the context
        cl::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        // Create a command queue and use the first device
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // Read source file
        auto sourceCode = oclReadSourcesFromFile("reduce_kernel.cl");
        cl::Program::Sources sources(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

        // Make program of the source code in the context
        cl::Program program(context, sources);

        // Build program for these specific devices
        try {
            program.build(devices);
        }
        catch (cl::Error error) {
            oclPrintError(error);
            // Detailed build errors:
            std::cerr << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
            std::cerr << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]) << std::endl;
            std::cerr << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
            throw error;
        }
//
//        // Make kernel
//        Kernel kernel_0(program, "reduce"), kernel_1(program, "reduce");
//#pragma endregion
//
//        for (unsigned size = 1 << 20; size <= 1 << 27; size = size << 1) {
//            std::vector<int> input(size);
//            std::default_random_engine rand(1);
//            std::generate(input.begin(), input.end(), [&]() {return rand() % 512 - 256; });
//
//            auto CPU_result = std::accumulate(input.begin(), input.end(), 0);
//
//#pragma region Execute kernel
//            // Create memory buffers
//            Buffer io_buffer_0(context, CL_MEM_READ_WRITE, size * sizeof(int));
//            Buffer io_buffer_1(context, CL_MEM_READ_WRITE, size * sizeof(int));
//
//            // Copy input to the memory buffer
//            queue.enqueueWriteBuffer(io_buffer_0, CL_TRUE, 0, size * sizeof(int), input.data());
//
//            cl_ulong round = 0;
//            const unsigned GROUP_SIZE = 128;
//
//            kernel_0.setArg(0, io_buffer_0);
//            kernel_0.setArg(1, io_buffer_1);
//            kernel_0.setArg(2, GROUP_SIZE * sizeof(int), nullptr);
//
//            kernel_1.setArg(0, io_buffer_1);
//            kernel_1.setArg(1, io_buffer_0);
//            kernel_1.setArg(2, GROUP_SIZE * sizeof(int), nullptr);
//
//            double time = 0;
//            for (unsigned rem_size = size; rem_size > 1; rem_size = round_up_div(rem_size, GROUP_SIZE), ++round) {
//
//                // Run the kernel on specific ND range
//                Event operation;
//                int t1 = round_up_div(rem_size, GROUP_SIZE) * GROUP_SIZE;
//                queue.enqueueNDRangeKernel(round % 2 == 0 ? kernel_0 : kernel_1,
//                    NullRange, t1, GROUP_SIZE,
//                    nullptr, &operation);
//
//                time += oclGetTiming(operation);
//            }
//
//            queue.finish();
//
//            // Read buffer into a local variable
//            int GPU_result;
//            queue.enqueueReadBuffer(round % 2 == 0 ? io_buffer_0 : io_buffer_1, CL_TRUE, 0, sizeof(int), &GPU_result);
//#pragma endregion
//
//            std::cout << time << std::endl;
//
//            if (CPU_result != GPU_result) {
//                std::cerr << "computation error" << std::endl;
//            }
//        }

    }
    catch (cl::Error error) {
        oclPrintError(error);
    }

    std::cin.get();

    return 0;
}

void read_input(const std::string& data_file, parameters& p, std::vector<float>& IPs, std::vector<float>& Fs, std::vector<float>& Bs, std::vector<float>& OPs)
{
    TCHAR buffer[MAX_PATH] = { 0 };
    GetModuleFileName(NULL, buffer, MAX_PATH);
    FILE* f = fopen(data_file.c_str(), "rb");
    if (f == 0) {
        std::cerr << "Cannot open data file: " << data_file << std::endl << "Errno: " << std::strerror(errno) << std::endl;
        exit(-1);
    }
    fread((char*)&p.IP_N, sizeof(p.IP_N), 1, f);
    fread((char*)&p.IP_w, sizeof(p.IP_w), 1, f);
    fread((char*)&p.IP_h, sizeof(p.IP_h), 1, f);
    IPs.resize(p.IP_N * p.IP_w * p.IP_h);
    fread((char*)IPs.data(), IPs.size() * sizeof(float), 1, f);
    p.IPs = IPs.data();

    fread((char*)&p.F_N, sizeof(p.F_N), 1, f);
    fread((char*)&p.F_w, sizeof(p.F_w), 1, f);
    fread((char*)&p.F_h, sizeof(p.F_h), 1, f);
    fread((char*)&p.F_d, sizeof(p.F_d), 1, f);
    Fs.resize(p.F_N * p.F_w * p.F_h * p.F_d);
    fread((char*)Fs.data(), Fs.size() * sizeof(float), 1, f);
    p.Fs = Fs.data();

    Bs.resize(p.F_N);
    fread((char*)Bs.data(), Bs.size() * sizeof(float), 1, f);
    p.Bs = Bs.data();

    fread((char*)&p.S_w, sizeof(p.S_w), 1, f);
    fread((char*)&p.S_h, sizeof(p.S_h), 1, f);

    p.OP_N = p.F_N;
    p.OP_w = (p.IP_w - p.F_w) / p.S_w + 1;
    p.OP_h = (p.IP_h - p.F_h) / p.S_h + 1;
    OPs.resize(p.OP_N * p.OP_w * p.OP_h);
    p.OPs = OPs.data();

    fclose(f);
}

void read_reference_output(const std::string& ref_fname, std::vector<float>& OPs)
{
    FILE* f = fopen("dat/reference_output.dat", "rb");

    if (f == 0) {
        std::cerr << "Cannot open reference file: " << ref_fname << std::endl;
        exit(-1);
    }

    unsigned rOP_N, rOP_w, rOP_h;
    fread(&rOP_N, sizeof(rOP_N), 1, f);
    fread(&rOP_w, sizeof(rOP_w), 1, f);
    fread(&rOP_h, sizeof(rOP_h), 1, f);
    OPs.resize(rOP_N * rOP_w * rOP_h);
    fread((char*)OPs.data(), OPs.size() * sizeof(float), 1, f);
    fclose(f);
}

float validate_result(const std::vector<float>& refOPs, const std::vector<float>& OPs)
{
    assert(refOPs.size() == OPs.size() && "Wrong output size!");
    float error = 0;
    for (size_t i = 0; i < refOPs.size(); ++i) {
        error += std::fabs(refOPs[i] - OPs[i]);
    }
    return error;
}
