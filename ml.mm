#define NS_STRING(s) NS::String::string(s, NS::ASCIIStringEncoding)
#define NS_ERR(ns_err) err->localizedDescription()->cString(NS::ASCIIStringEncoding)
#define MTL_ERR(msg, err)                                                                          \
    std::cerr << "(Metal) " << msg << NS_ERR(err) << std::endl;                                    \
    exit(-1);

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

ML* ML::init() {
    ML* ml = new ML {};
    NS::Error* err = nullptr;
    ml->device = MTL::CreateSystemDefaultDevice();
    ml->queue = ml->device->newCommandQueue();
    if (!ml->queue) {
        std::cerr << "Failed to create command queue" << std::endl;
        exit(-1);
    }
    const NS::String* kernel_path = NS_STRING("build/kernels.metallib");
    ml->library = ml->device->newLibrary(kernel_path, &err);
    if (err != nullptr) {
        MTL_ERR("(Metal) Failed to create a MTLLibrary: ", err);
    }

    return ml;
}

Tensor::Tensor(
    size_t n_dim, uint64_t dims[MAX_DIMS], size_t data_size, float* data, MTL::Device* device
)
    : n_dim(n_dim) {
    memcpy(this->dims, dims, n_dim * sizeof(uint64_t));
    this->data = device->newBuffer(
        data, data_size, MTL::ResourceStorageModeShared, ^(void* buf, NS::UInteger) { free(buf); }
    );
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

Tensor* Tensor::zeros(uint64_t d0, uint64_t d1, uint64_t d2, MTL::Device* device) {
    float* data = (float*)calloc(d0 * d1 * d2, sizeof(float));
    uint64_t dims[MAX_DIMS] = { d0, d1, d2, 0 };
    return new Tensor(3, dims, d0 * d1 * d2 * sizeof(float), data, device);
}

template <typename Rng> void fill_randn(float* buf, size_t n, Rng& rng) {
    std::normal_distribution<float> d { 0.0, 1.0 };
    for (size_t i = 0; i < n; ++i)
        buf[i] = d(rng);
}

template <typename Rng> Tensor* Tensor::randn(uint64_t d0, MTL::Device* device, Rng& rng) {
    Tensor* t = zeros(d0, device);
    fill_randn((float*)t->data->contents(), t->n_elem(), rng);
    return t;
}

template <typename Rng>
Tensor* Tensor::randn(uint64_t d0, uint64_t d1, MTL::Device* device, Rng& rng) {
    Tensor* t = zeros(d0, d1, device);
    fill_randn((float*)t->data->contents(), t->n_elem(), rng);
    return t;
}

template <typename Rng>
Tensor* Tensor::randn(uint64_t d0, uint64_t d1, uint64_t d2, MTL::Device* device, Rng& rng) {
    Tensor* t = zeros(d0, d1, d2, device);
    fill_randn((float*)t->data->contents(), t->n_elem(), rng);
    return t;
}

template Tensor* Tensor::randn(uint64_t d0, MTL::Device* device, std::mt19937& rng);
template Tensor* Tensor::randn(uint64_t d0, uint64_t d1, MTL::Device* device, std::mt19937& rng);
template Tensor*
Tensor::randn(uint64_t d0, uint64_t d1, uint64_t d2, MTL::Device* device, std::mt19937& rng);

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

nn::Linear::Linear(
    uint64_t n_input,
    uint64_t n_output,
    const Tensor& weight,
    const Tensor* bias,
    MTL::Device* device
)
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

MambaBlock::MambaBlock(const ML& ml) {
    NS::Error* err = nullptr;
    MTL::Function* discretize_fn = ml.library->newFunction(NS_STRING("discretize"), nullptr, &err);
    if (err != nullptr) {
        MTL_ERR("Failed to create shader function <discretize>: ", err);
    }
    this->discretizePSO = ml.device->newComputePipelineState(discretize_fn, &err);
    if (err != nullptr) {
        MTL_ERR("Failed to create PSO for shader function <discretize>: ", err);
    }
}

void MambaBlock::discretize(
    const Tensor& u,
    const Tensor& delta,
    const Tensor& A,
    const Tensor& B,
    Tensor& deltaA,
    Tensor& deltaB_u,
    const ML& ml
) {
    int32_t l = u.dims[0];
    int32_t d = u.dims[1];
    int32_t n = A.dims[1];

    MTL::CommandBuffer* cmd_buf = ml.queue->commandBuffer();
    MTL::ComputeCommandEncoder* cmd_enc = cmd_buf->computeCommandEncoder();
    cmd_enc->setComputePipelineState(discretizePSO);
    cmd_enc->setBytes(&d, sizeof(int32_t), 0);
    cmd_enc->setBytes(&n, sizeof(int32_t), 1);
    cmd_enc->setBuffer(u.data, 0, 2);
    cmd_enc->setBuffer(delta.data, 0, 3);
    cmd_enc->setBuffer(A.data, 0, 4);
    cmd_enc->setBuffer(B.data, 0, 5);
    cmd_enc->setBuffer(deltaA.data, 0, 6);
    cmd_enc->setBuffer(deltaB_u.data, 0, 7);
    MTL::Size threads_per_grid = MTL::Size::Make(l, d, 1);
    NS::UInteger thread_group_w = discretizePSO->threadExecutionWidth();
    NS::UInteger thread_group_h = discretizePSO->maxTotalThreadsPerThreadgroup() / thread_group_w;
    MTL::Size threads_per_thread_group = MTL::Size::Make(thread_group_w, thread_group_h, 1);
    cmd_enc->dispatchThreads(threads_per_grid, threads_per_thread_group);
    cmd_enc->endEncoding();

    cmd_buf->commit();
    cmd_buf->waitUntilCompleted();
}
