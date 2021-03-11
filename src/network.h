#ifndef __NETWORK_H
#define __NETWORK_H

/**
 * network class
 * */

#include "./utils.h"
#include "./layer.h"
#include "./loss.h"
#include "./optimizer.h"
#include <vector>

class Network {
    private:
        std::vector<Layer*> layers;
        Loss* loss;
        
    public:
        Network() : loss(NULL) {}
        ~Network() {
            for (int i = 0; i < layers.size(); i++) {
                delete layers[i];
            }
            if (loss) {
                delete loss;
            }
        }

        void add_layer(Layer* layer) { layers.push_back(layer); }
        void add_loss(Loss* loss_function) { loss = loss_function; }
        
        void forward(const Matrix& input);
        void backward(const Matrix& input, const Matrix& target);
        void update(Optimizer& opt);

        const Matrix output() { return layers.back()->output(); }
        double get_loss() { return loss->output(); }

        std::vector<std::vector<double>> get_parameters() const;
        void set_parameters(const std::vector<std::vector<double>>& parameters);
        std::vector<std::vector<double>> get_derivatives() const;
        void debug(const Matrix& input, const Matrix& target, int n, int seed = -1);
}; 

#endif