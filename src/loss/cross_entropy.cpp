#include "./cross_entropy.h"

void CrossEntropy::evaluate(const Matrix &pred, const Matrix &target) {
    /**
     * this function can only be used on two-classfication problem
     * */
    const int n = pred.cols();
    // forward
    // make the label's dimension same as the pred dimension
    Matrix tmp(2, n);
    Matrix another_label = Matrix::Ones(1, n) - target;
    tmp << another_label,
           target;
    // calculate loss: -\sum{label_matrix * log(prediction_matrix)}
    loss = - (tmp.array().cwiseProduct(pred.array().log())).sum();
    loss /= n;
    // backward
    grad = - tmp.array().cwiseQuotient(pred.array()) / n;
}