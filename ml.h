#include <random>
#define MAX_DIMS 4

#include <cstdint>
#include <memory>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include <Metal/Metal.hpp>
#pragma clang diagnostic pop
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

class ML {
public:
    MTL::Device* device;
    MTL::CommandQueue* queue;

    static ML* init();

private:
    ML(MTL::Device* device, MTL::CommandQueue* queue);
};

class Tensor {
public:
    size_t n_dim;
    uint64_t dims[MAX_DIMS];
    MTL::Buffer* data;

    ~Tensor();
    static Tensor* zeros(uint64_t d0, MTL::Device* device);
    static Tensor* zeros(uint64_t d0, uint64_t d1, MTL::Device* device);
    template <typename Rng> static Tensor* randn(uint64_t d0, MTL::Device* device, Rng& rng);
    template <typename Rng>
    static Tensor* randn(uint64_t d0, uint64_t d1, MTL::Device* device, Rng& rng);
    size_t n_elem() const;
    size_t buf_size() const;
    id<MTLBuffer> buf_id() const;

    friend std::ostream& operator<<(std::ostream& os, Tensor const& t);

private:
    Tensor(
        size_t n_dim, uint64_t dims[MAX_DIMS], size_t data_size, float* data, MTL::Device* device);
};

namespace nn {
class Linear {
public:
    uint64_t n_input;
    uint64_t n_output;
    MPSMatrix* weight;
    const Tensor* bias;

    Linear(uint64_t n_input, uint64_t n_output, const Tensor& weight, const Tensor* bias,
        MTL::Device* device);
    ~Linear();

    void forward(const Tensor& x, Tensor& y, ML& ml) const;

private:
    MPSMatrixVectorMultiplication* kernel;
    MPSVectorDescriptor* input_vec_desc;
    MPSVectorDescriptor* output_vec_desc;
};
}
