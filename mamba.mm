#include "ml.h"
#include <iostream>
#include <memory>

int main() {
    std::seed_seq seed { 0 };
    std::mt19937 rng(seed);

    ML* ml = ML::init();

    uint64_t n_input = 3;
    uint64_t n_output = 2;

    Tensor* weight = Tensor::randn(n_output, n_input, ml->device, rng);
    Tensor* bias = Tensor::randn(n_output, ml->device, rng);
    std::cout << "weight: \n" << *weight << std::endl;
    std::cout << "bias: \n" << *bias << std::endl;
    nn::Linear* layer = new nn::Linear(n_input, n_output, *weight, bias, ml->device);

    Tensor* x = Tensor::randn(n_input, ml->device, rng);
    Tensor* y = Tensor::zeros(n_output, ml->device);

    layer->forward(*x, *y, *ml);
    std::cout << "x: \n" << *x << std::endl;
    std::cout << "y: \n" << *y << std::endl;
    return 0;
}
