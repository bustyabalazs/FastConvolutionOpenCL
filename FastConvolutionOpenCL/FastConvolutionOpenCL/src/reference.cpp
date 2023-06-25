#include <algorithm>
#include "../inc/convolution.h"

inline float NonLin(float v)
{
    // Sigmoid
    // return 1 / (1 + exp(-v));
    // RELU
    return std::max(0.0f, v);
    // The nonlinear function is a simle sign in the binarized case
    // return std::signbit(v);
}

struct convolution
{
    convolution(parameters* p) : p(p) {}
    parameters* p;
};

// The reference CPU implementation
struct convolution* convolution_new(struct parameters* p)
{
    return new convolution(p);
}

// Reference CPU implementation
void convolution_run(struct convolution* c)
{
    struct parameters& p = *c->p;

    // Determine boundaries
    unsigned int y_start = 0;
    unsigned int y_stop = p.IP_h - p.F_h + 1;
    unsigned int x_start = 0;
    unsigned int x_stop = p.IP_w - p.F_w + 1;

    // PART A: Convolution for the input planes
    for (unsigned int y0 = y_start; y0 < y_stop; y0 = y0 + p.S_h) {
        for (unsigned int x0 = x_start; x0 < x_stop; x0 = x0 + p.S_w) {
            for (unsigned int f = 0; f < p.F_N; ++f) {
                // The result of the convolution
                float value = 0;

                // Calculate the convolution for the x0, y0 point and f filter
                for (unsigned int d = 0; d < p.F_d; ++d) {
                    for (unsigned int y = 0; y < p.F_h; ++y) {
                        for (unsigned int x = 0; x < p.F_w; ++x) {
                            unsigned int ip_idx = d * p.IP_w * p.IP_h + (y0 + y) * p.IP_w + x0 + x;
                            unsigned int f_idx = f * p.F_d * p.F_h * p.F_w + d * p.F_h * p.F_w + y * p.F_w + x;
                            value += p.IPs[ip_idx] * p.Fs[f_idx];
                        }
                    }
                }

                // Write out the result of the convolution into the output plane
                unsigned int op_idx = f * p.OP_h * p.OP_w + y0 / p.S_h * p.OP_w + x0 / p.S_w;
                p.OPs[op_idx] = value + p.Bs[f];
            }
        }
    }

    // PART B: Applying the nonlinear function pointwise to each output plane
    for (unsigned int o = 0; o < p.OP_N; ++o) {
        for (unsigned int y = 0; y < p.OP_h; ++y) {
            for (unsigned int x = 0; x < p.OP_w; ++x) {
                unsigned int idx = o * p.OP_w * p.OP_h + y * p.OP_w + x;
                p.OPs[idx] = NonLin(p.OPs[idx]);
            }
        }
    }

#ifdef WITH_PART_C
    // PART C: Do the max polling of each output layer and write back the result to the same buffer
    // Not included into the benchmark it is here only for completness of the code
    for (unsigned int o = 0; o < p.F_N; ++o) {
        for (unsigned int y0 = 0; y0 < p.OP_h - p.MP_h / 2; y0 = y0 + p.MP_h) {
            for (unsigned int x0 = 0; x0 < p.OP_w - p.MP_w / 2; x0 = x0 + p.MP_w) {
                float value = 0;

                for (unsigned int y = 0; y < p.MP_h; ++y) {
                    for (unsigned int x = 0; x < p.MP_w; ++x) {
                        unsigned int op_idx = o * p.OP_w * p.OP_h + (y0 + y) * p.OP_w + x0 + x;
                        if (p.OPs[op_idx] > value)
                            value = p.OPs[op_idx];
                    }
                }

                unsigned int out_idx = o * p.OP_w * p.OP_h / p.MP_w / p.MP_h + y0 / p.MP_h * p.OP_w / p.MP_w + x0 / p.MP_w;
                p.OPs[out_idx] = value;
            }
        }
    }
#endif
}

void convolution_result(struct convolution* c)
{
    // Here can GPU implementation copy result back to host.
    // Reference implementation already have output in OPs
}

void convolution_destroy(struct convolution* c)
{
    // delete the convolution
    delete c;
}
