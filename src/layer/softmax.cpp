#include "./softmax.h"

void Softmax::forward(const Matrix &bottom) {
    top.array() = (bottom.rowwise() - bottom.colwise().maxCoeff()).array().exp();
    RowVector z_exp_sum = top.colwise().sum();  // \sum{ exp(z) }
    top.array().rowwise() /= z_exp_sum;
}

void Softmax::backward(const Matrix &bottom, const Matrix &grad_top) {
    RowVector temp_sum = top.cwiseProduct(grad_top).colwise().sum();
    grad.array() = top.array().cwiseProduct(grad_top.array().rowwise() - temp_sum);
}

