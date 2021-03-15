#include "./network.h"
#include <cstdlib>
#include <stdexcept>
#include <vector>

void Network::forward(const Matrix &input) {
    if (layers.empty()) return;
    layers[0]->forward(input);

    for (int i = 1; i < layers.size(); i++) {
        layers[i]->forward(layers[i-1]->output());
    }
}

void Network::backward(const Matrix &input, const Matrix &target) {
    int layer_num = layers.size();

    if (layer_num == 0) return ;

    loss->evaluate(layers[layer_num-1]->output(), target);
    if(layer_num == 1) {
        layers[0]->backward(input, loss->back_gradient());
        return;
    }
    layers[layer_num-1]->backward(layers[layer_num-2]->output(), 
                                    loss->back_gradient());
    for (int i = layer_num - 2; i > 0; i--) {
        layers[i]->backward(layers[i-1]->output(), layers[i+1]->back_gradient());
    }
    layers[0]->backward(input, layers[1]->back_gradient());
}

void Network::update(Optimizer &opt) {
    for (int i = 0; i < layers.size(); i++) {
        layers[i]->update(opt);
    }
}

std::vector<std::vector<double>> Network::get_parameters() const {
    const int layer_num = layers.size();
    std::vector<std::vector<double>> res;
    res.reserve(layer_num);
    for (int i = 0; i < layer_num; i++) {
        res.push_back(layers[i]->get_parameters());
    }
    return res;
}

void Network::set_parameters(const std::vector<std::vector<double>>& parameters) {
    const int layer_num = layers.size();
    if (static_cast<int>(parameters.size()) != layer_num) {
        std::invalid_argument("layer size doesn't match");
    }
    for (int i = 0; i < layer_num; i++) {
        layers[i]->set_parameters(parameters[i]);
    }
}

std::vector<std::vector<double>> Network::get_derivatives() const {
    const int layer_num = layers.size();
    std::vector<std::vector<double>> res;
    res.reserve(layer_num);
    for (int i = 0; i < layer_num; i++) {
        res.push_back(layers[i]->get_derivatives());
    }
    return res;   
}

void Network::debug(const Matrix &input, const Matrix &target, int n, int seed) {
    if (seed > 0) std::srand(seed);

    this->forward(input);
    this->backward(input, target);
    std::vector<std::vector<double>> parameters = this->get_parameters();
    std::vector<std::vector<double>> derivatives = this->get_derivatives();

    const double eps = 1e-4;
    const int layer_num = derivatives.size();
    for (int i = 0; i < n; i++) {
        // randomly choose a layer
        const int layer_id = int(std::rand() / double(RAND_MAX) * layer_num);
        const int parameter_num = parameters[layer_id].size();
        // check whether the layer already has parameter
        if (parameter_num < 1) continue;
        // randomly choose a parameter
        const int parameter_id = int(std::rand() / double(RAND_MAX) * parameter_num);
        const double old_para = parameters[layer_id][parameter_id];

        // change a little
        parameters[layer_id][parameter_id] -= eps;
        this->set_parameters(parameters);
        this->forward(input);
        this->backward(input, target);
        const double loss_pre = loss->output();

        parameters[layer_id][parameter_id] += (2 * eps);
        this->set_parameters(parameters);
        this->forward(input);
        this->backward(input, target);
        const double loss_post = loss->output();

        const double deriv_est = (loss_post - loss_pre) / (2 * eps);

        std::cout << "[layer " << layer_id << ", parameter " << parameter_id <<
        "] derivative = " << derivatives[layer_id][parameter_id] << ", est = " << deriv_est <<
        ", diff = " << (deriv_est - derivatives[layer_id][parameter_id]) << std::endl;

        parameters[layer_id][parameter_id] = old_para;
    }
    this->set_parameters(parameters);
}