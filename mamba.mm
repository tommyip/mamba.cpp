#include "ml.h"
#include <iostream>
#include <memory>

int main() {
    std::seed_seq seed { 0 };
    std::mt19937 rng(seed);

    ML* ml = ML::init();

    const uint64_t l = 3;
    const uint64_t d_in = 5;
    const uint64_t n = 2;

    const Tensor* u = Tensor::randn(l, d_in, ml->device, rng);
    const Tensor* delta = Tensor::randn(l, d_in, ml->device, rng);
    const Tensor* A = Tensor::randn(d_in, n, ml->device, rng);
    const Tensor* B = Tensor::randn(l, n, ml->device, rng);
    Tensor* deltaA = Tensor::zeros(l, d_in, ml->device);
    Tensor* deltaB_u = Tensor::zeros(l, d_in, ml->device);

    MambaBlock* layer = new MambaBlock(*ml);
    layer->discretize(*u, *delta, *A, *B, *deltaA, *deltaB_u, *ml);

    std::cout << *deltaA << std::endl;

    return 0;
}
