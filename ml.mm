#include "ml.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <Metal/Metal.hpp>
#pragma clang diagnostic pop

ML::ML(MTL::Device* device, MTL::CommandQueue* queue)
    : device(device)
    , queue(queue) { }

ML* ML::init() {
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue* queue = device->newCommandQueue();
    if (!queue) {
        std::cout << "(Metal) Failed to create command queue" << std::endl;
    }
    return new ML(device, queue);
}

Tensor::Tensor(
    size_t n_dim, uint64_t dims[MAX_DIMS], size_t data_size, float* data, MTL::Device* device)
    : n_dim(n_dim) {
    memcpy(this->dims, dims, n_dim * sizeof(uint64_t));
    this->data = device->newBuffer(
        data, data_size, MTL::ResourceStorageModeShared, ^(void* buf, NS::UInteger) { free(buf); });
}

Tensor::~Tensor() { data->release(); }

Tensor* Tensor::zeros(uint64_t d0, MTL::Device* device) {
    float* data = (float*)calloc(d0, sizeof(float));
    uint64_t dims[MAX_DIMS] = { d0, 0, 0, 0 };
    return new Tensor(1, dims, d0 * sizeof(float), data, device);
}

Tensor* Tensor::zeros(uint64_t d0, uint64_t d1, MTL::Device* device) {
    float* data = (float*)calloc(d0 * d1, sizeof(float));
    uint64_t dims[MAX_DIMS] = { d0, d1, 0, 0 };
    return new Tensor(2, dims, d0 * d1 * sizeof(float), data, device);
}

template <typename Rng> void fill_randn(float* buf, size_t n, Rng& rng) {
    std::normal_distribution<float> d { 0.0, 1.0 };
    for (size_t i = 0; i < n; ++i)
        buf[i] = d(rng);
}

template <typename Rng> Tensor* Tensor::randn(uint64_t d0, MTL::Device* device, Rng& rng) {
    Tensor* t = zeros(d0, device);
    fill_randn((float*)t->data->contents(), d0, rng);
    return t;
}

template <typename Rng>
Tensor* Tensor::randn(uint64_t d0, uint64_t d1, MTL::Device* device, Rng& rng) {
    Tensor* t = zeros(d0, d1, device);
    fill_randn((float*)t->data->contents(), d0 * d1, rng);
    return t;
}

template Tensor* Tensor::randn(uint64_t d0, MTL::Device* device, std::mt19937& rng);
template Tensor* Tensor::randn(uint64_t d0, uint64_t d1, MTL::Device* device, std::mt19937& rng);

size_t Tensor::n_elem() const {
    size_t n = dims[0];
    for (size_t i = 1; i < n_dim; ++i)
        n *= dims[i];
    return n;
}

size_t Tensor::buf_size() const { return n_elem() * sizeof(float); }

inline id<MTLBuffer> Tensor::buf_id() const { return (__bridge id<MTLBuffer>)data; }

std::ostream& operator<<(std::ostream& os, Tensor const& t) {
    float* data = (float*)t.data->contents();
    os << "[";
    switch (t.n_dim) {
    case 1: {
        for (uint64_t i = 0; i < t.dims[0]; ++i) {
            if (i > 0)
                os << " ";
            os << data[i];
        }
        break;
    }
    case 2: {
        for (uint64_t j = 0; j < t.dims[0]; ++j) {
            if (j > 0)
                os << "\n ";
            os << "[";
            for (uint64_t i = 0; i < t.dims[1]; ++i) {
                if (i > 0)
                    os << " ";
                os << data[t.dims[1] * j + i];
            }
            os << "]";
        }
        break;
    }
    }
    os << "]";
    return os;
}

nn::Linear::Linear(uint64_t n_input, uint64_t n_output, const Tensor& weight, const Tensor* bias,
    MTL::Device* device)
    : n_input(n_input)
    , n_output(n_output)
    , bias(bias) {
    assert(weight.n_dim == 2 && weight.dims[0] == n_output && weight.dims[1] == n_input);
    assert(bias->n_dim == 1 && bias->dims[0] == n_output);

    id<MTLBuffer> buf_weight = weight.buf_id();
    [buf_weight retain];
    MPSMatrixDescriptor* desc_weight =
        [MPSMatrixDescriptor matrixDescriptorWithRows:weight.dims[0]
                                              columns:weight.dims[1]
                                             rowBytes:weight.dims[1] * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    this->weight = [[MPSMatrix alloc] initWithBuffer:buf_weight descriptor:desc_weight];

    this->input_vec_desc = [MPSVectorDescriptor vectorDescriptorWithLength:n_input
                                                                  dataType:MPSDataTypeFloat32];
    this->output_vec_desc = [MPSVectorDescriptor vectorDescriptorWithLength:n_output
                                                                   dataType:MPSDataTypeFloat32];

    this->kernel =
        [[MPSMatrixVectorMultiplication alloc] initWithDevice:(__bridge id<MTLDevice>)device
                                                    transpose:false
                                                         rows:n_output
                                                      columns:n_input
                                                        alpha:1.0
                                                         beta:1.0];
}

nn::Linear::~Linear() { [weight.data release]; }

/**
 * y = Ax + b
 * where
 *   A : weight
 *   b : bias
 */
void nn::Linear::forward(const Tensor& x, Tensor& y, ML& ml) const {
    assert(x.n_dim == 1 && x.dims[0] == n_input);
    assert(y.n_dim == 1 && y.dims[0] == n_output);

    // The matrix vector multiplication kernel adds to the result vector
    // so we first set the result as the bias.
    if (bias != nullptr) {
        memcpy(y.data->contents(), bias->data->contents(), bias->buf_size());
    } else {
        memset(y.data->contents(), 0, bias->buf_size());
    }

    MPSVector* input = [[MPSVector alloc] initWithBuffer:x.buf_id()
                                              descriptor:this->input_vec_desc];
    MPSVector* result = [[MPSVector alloc] initWithBuffer:y.buf_id()
                                               descriptor:this->output_vec_desc];

    MTL::CommandBuffer* cmd_buf = ml.queue->commandBuffer();
    assert(cmd_buf != nullptr);

    [kernel encodeToCommandBuffer:(__bridge id<MTLCommandBuffer>)cmd_buf
                      inputMatrix:weight
                      inputVector:input
                     resultVector:result];
    cmd_buf->commit();
    cmd_buf->waitUntilCompleted();

    [input release];
    [result release];
}
