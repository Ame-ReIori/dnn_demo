#include "./cross_entropy.h"

void CrossEntropy::evaluate(const Matrix &pred, const Matrix &target) {
    const int n = pred.cols();
    const double eps = 1e-8;
    // forward
    loss = - (target.array().cwiseProduct((pred.array() + eps).log())).sum();
    loss /= n;
    // backward
    grad = - target.array().cwiseQuotient(pred.array() + eps) / n;
}