#include "./mse.h"

void MSE::evaluate(const Matrix &pred, const Matrix &target) {
    const int n = pred.cols();

    //forward
    Matrix diff = pred - target;
    loss = diff.cwiseProduct(diff).sum();
    loss /= n;
    //backward
    grad = diff * 2 / n;
}