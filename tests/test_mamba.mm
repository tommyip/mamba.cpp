#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <utility>
extern "C" {
#include "subprojects/gguf-tools/gguflib.h"
}
#include "ml.h"

using Tensors = std::map<std::string, Tensor*>;

Tensors gguf_read(gguf_ctx* ctx, MTL::Device* device) {
    gguf_skip_key_values_section(ctx);
    std::map<std::string, Tensor*> tensors;
    gguf_tensor gg_tensor;
    while (gguf_get_tensor(ctx, &gg_tensor)) {
        assert(gg_tensor.type == GGUF_TYPE_F32);
        std::string name(gg_tensor.name, gg_tensor.namelen);
        Tensor* tensor = Tensor::from_gguf(&gg_tensor, device);
        tensors.insert(std::pair(name, tensor));
    }
    return tensors;
}

void gguf_write(const char* filename, Tensors& tensors) {
    gguf_ctx* ctx = gguf_create(filename, GGUF_OVERWRITE);

    uint64_t offset = 0;
    for (auto it = tensors.cbegin(); it != tensors.cend(); ++it) {
        Tensor* tensor = it->second;
        uint64_t dims[GGUF_TENSOR_MAX_DIM];
        std::reverse_copy(
            std::begin(tensor->dims), std::begin(tensor->dims) + tensor->n_dim, std::begin(dims)
        );
        gguf_append_tensor_info(
            ctx, it->first.c_str(), it->first.length(), tensor->n_dim, dims, GGUF_TYPE_F32, offset
        );
        offset += tensor->buf_size();
        offset += gguf_get_alignment_padding(ctx->alignment, offset);
    }
    for (auto it = tensors.cbegin(); it != tensors.cend(); ++it) {
        gguf_append_tensor_data(ctx, it->second->data->contents(), it->second->buf_size());
    }
    gguf_close(ctx);
}

Tensors test_sanity(Tensors& in, ML&) { return { { "y", in["x"] } }; }

Tensors test_linear_layer(Tensors& in, ML& ml) {
    const Tensor* weight = in["weight"];
    const Tensor* bias = in["bias"];
    const uint64_t n_input = weight->dims[1];
    const uint64_t n_output = weight->dims[0];
    const nn::Linear linear(n_input, n_output, *weight, bias, ml.device);

    const Tensor* x = in["x"];
    Tensor* y = Tensor::zeros(n_output, ml.device);
    linear.forward(*x, *y, ml);

    return { { "y", y } };
}

Tensors test_mamba_block_discretize(Tensors& in, ML& ml) {
    const Tensor* u = in["u"];
    const Tensor* delta = in["delta"];
    const Tensor* A = in["A"];
    const Tensor* B = in["B"];
    uint64_t l = u->dims[0];
    uint64_t d = u->dims[1];
    uint64_t n = A->dims[1];
    Tensor* deltaA = Tensor::zeros(l, d, n, ml.device);
    Tensor* deltaB_u = Tensor::zeros(l, d, n, ml.device);
    MambaBlock block(ml);
    block.discretize(*u, *delta, *A, *B, *deltaA, *deltaB_u, ml);
    return { { "deltaA", deltaA }, { "deltaB_u", deltaB_u } };
}

const std::map<std::string, Tensors (*)(Tensors&, ML&)> TEST_FNS
    = { { "sanity", &test_sanity },
        { "linear_layer", &test_linear_layer },
        { "mamba_block_discretize", &test_mamba_block_discretize } };

int main(int argc, char* argv[]) {
    assert(argc == 4);
    ML* ml = ML::init();
    std::string test_fn_name(argv[1]);
    const char* path_in = argv[2];
    const char* path_out = argv[3];

    auto test_fn_it = TEST_FNS.find(test_fn_name);
    if (test_fn_it == TEST_FNS.end()) {
        fprintf(stderr, "test function <%s> not found", test_fn_name.c_str());
        exit(-1);
    }
    auto test_fn = test_fn_it->second;

    gguf_ctx* gguf_in = gguf_open(path_in);
    Tensors tensors_in = gguf_read(gguf_in, ml->device);

    Tensors tensors_out = test_fn(tensors_in, *ml);

    gguf_write(path_out, tensors_out);
    gguf_close(gguf_in);
}
