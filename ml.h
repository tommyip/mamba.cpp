#define MAX_DIMS 4

#include <Metal/Metal.hpp>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstdint>
#include <memory>
#include <random>
extern "C" {
#include "subprojects/gguf-tools/gguflib.h"
}

class ML {
public:
    MTL::Device* device;
    MTL::CommandQueue* queue;
    MTL::Library* library;

    static ML* init();
};

class Tensor {
public:
    size_t n_dim;
    uint64_t dims[MAX_DIMS];
    MTL::Buffer* data;

    ~Tensor();
    static Tensor* zeros(uint64_t d0, MTL::Device* device);
    static Tensor* zeros(uint64_t d0, uint64_t d1, MTL::Device* device);
    static Tensor* zeros(uint64_t d0, uint64_t d1, uint64_t d2, MTL::Device* device);
    template <typename Rng> static Tensor* randn(uint64_t d0, MTL::Device* device, Rng& rng);
    template <typename Rng>
    static Tensor* randn(uint64_t d0, uint64_t d1, MTL::Device* device, Rng& rng);
    template <typename Rng>
    static Tensor* randn(uint64_t d0, uint64_t d1, uint64_t d2, MTL::Device* device, Rng& rng);
    static Tensor* from_gguf(gguf_tensor* tensor, MTL::Device* device);
    size_t n_elem() const;
    size_t buf_size() const;
    id<MTLBuffer> buf_id() const;

    friend std::ostream& operator<<(std::ostream& os, Tensor const& t);

private:
    Tensor(
        size_t n_dim, uint64_t dims[MAX_DIMS], size_t data_size, float* data, MTL::Device* device
    );
};

namespace nn {
class Linear {
public:
    uint64_t n_input;
    uint64_t n_output;
    MPSMatrix* weight;
    const Tensor* bias;

    Linear(
        uint64_t n_input,
        uint64_t n_output,
        const Tensor& weight,
        const Tensor* bias,
        MTL::Device* device
    );
    ~Linear();

    void forward(const Tensor& x, Tensor& y, ML& ml) const;

private:
    MPSMatrixVectorMultiplication* kernel;
    MPSVectorDescriptor* input_vec_desc;
    MPSVectorDescriptor* output_vec_desc;
};
}

class MambaBlock {
public:
    MambaBlock(const ML& ml);

private:
    const MTL::ComputePipelineState* discretizePSO;

public:
    /**
     * Discretize continuous parameters
     *
     * deltaA = exp(einsum('ld, dn -> ldn', delta, A))
     * deltaB_u = einsum('ld, ln, ld -> ldn', delta, B, u)
     *
     * @param u input, shape (l, d_in)
     * @param delta input, shape (l, d_in)
     * @param A input, shape (d_in, n)
     * @param B input, shape (l, n)
     * @param deltaA output, shape (l, d_in)
     * @param deltaB_u output, shape (l, d_in)
     * @param ml Metal state
     */
    void discretize(
        const Tensor& u,
        const Tensor& delta,
        const Tensor& A,
        const Tensor& B,
        Tensor& deltaA,
        Tensor& deltaB_u,
        const ML& ml
    );
};
