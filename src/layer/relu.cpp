#include "./relu.h"

void RELU::forward(const Matrix &bottom) {
    top = bottom.cwiseMax(0.0);
}

void RELU::backward(const Matrix &bottom, const Matrix &grad_top) {
    // d(L)/d(z_i) = d(L)/d(a_i) * d(a_i)/d(z_i)
    //             = d(L)/d(a_i) * 1*(z_i>0)
    Matrix positive = (bottom.array() > 0.0).cast<double>();
    grad = grad_top.cwiseProduct(positive);
}