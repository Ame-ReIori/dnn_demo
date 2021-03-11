#include "./full_connect.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

void FullConnect::init() {
    weight.resize(dim_in, dim_out);
    bias.resize(dim_out);
    grad_weight.resize(dim_in, dim_out);
    grad_bias(dim_out);
    set_normal_random(weight.data(), weight.size(), 0, 0.1);
    set_normal_random(bias.data(), bias.size(), 0, 0.1);
}

void FullConnect::forward(const Matrix &bottom) {
    const int sample_num = bottom.cols();
    top.resize(dim_out, sample_num);
    top = weight.transpose() * bottom;
    top.colwise() += bias;
}

void FullConnect::backward(const Matrix &bottom, const Matrix &grad_top) {
    const int sample_num = bottom.cols();

    // calculate gradient
    // d(L)/d(w') = d(L)/d(z) * x'
    // d(L)/d(b) = \sum{ d(L)/d(z_i) }
    grad_weight = bottom * grad_top.transpose();
    grad_bias = grad_top.rowwise().sum();
    // d(L)/d(x) = w * d(L)/d(z)
    grad.resize(dim_out, sample_num);
    grad = weight * grad_top;
}

void FullConnect::update(Optimizer &opt) {
    Vector::AlignedMapType weight_vec(weight.data(), weight.size());
    Vector::AlignedMapType bias_vec(bias.data(), bias.size());
    Vector::ConstAlignedMapType grad_weight_vec(grad_weight.data(), grad_weight.size());
    Vector::ConstAlignedMapType grad_bias_vec(grad_bias.data(), grad_bias.size());

    opt.update(weight_vec, grad_weight_vec);
    opt.update(bias_vec, grad_bias_vec);
}

std::vector<double> FullConnect::get_parameters() const {
    std::vector<double> res(weight.size() + bias.size());
    std::copy(weight.data(), weight.data() + weight.size(), res.begin());
    std::copy(bias.data(), bias.data() + bias.size(), res.begin() + weight.size());
    return res;
}

std::vector<double> FullConnect::get_derivatives() const {
    std::vector<double> res(grad_weight.size() + grad_bias.size());
    std::copy(grad_weight.data(), grad_weight.data() + grad_weight.size(), res.begin());
    std::copy(grad_bias.data(), grad_bias.data() + grad_bias.size(), res.begin() + grad_weight.size());
}

void FullConnect::set_parameters(const std::vector<double> parameters) {
    if (static_cast<int>(parameters.size()) != weight.size() + bias.size()) {
        throw std::invalid_argument("size doesn't match");
    }
    std::copy(parameters.begin(), parameters.begin() + weight.size(), weight.data());
    std::copy(parameters.begin() + weight.size(), 
        parameters.begin() + weight.size() + bias.size(), bias.data());
}

