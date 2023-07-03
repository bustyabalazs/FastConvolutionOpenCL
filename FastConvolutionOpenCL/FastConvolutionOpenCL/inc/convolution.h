#pragma once

struct parameters
{
    // The input planes
    unsigned int IP_N;  // number of input planes
    unsigned int IP_w;  // width  of input planes;
    unsigned int IP_h;  // height of input planes;

    // The filters
    unsigned int F_N;  // number of filters;
    unsigned int F_w;  // width of filters;
    unsigned int F_h;  // height of filters;
    unsigned int F_d;  // depth of filters should be equal to number of input planes

    // Step sizes
    unsigned int S_w;  // Filter step size in width direction
    unsigned int S_h;  // Filter step size in height direction

    // The output planes
    unsigned int OP_N;  // Number of output planes
    unsigned int OP_w;  // Width  of output planes
    unsigned int OP_h;  // Height of output planes

    // The max poll size
    unsigned int MP_w;  // max_polling size in width direction
    unsigned int MP_h;  // Max polling size in height direction

    float* IPs;  // data for input planes
    float* Fs;   // data for filters
    float* Bs;
    float* OPs;  // data for output planes
};

// forward declaration
struct convolution;

extern "C" {
struct convolution* convolution_new(struct parameters* params);
void convolution_run(struct convolution* c);
void fast_convolution_run(struct convolution* c);
void convolution_result(struct convolution* c);
void convolution_destroy(struct convolution* c);
}

namespace Winograd{

    float* matrixMultiply(const float* A, const unsigned int rowsA, const float* B, const unsigned int columsB, const unsigned int columnsA_rowsB);

}
