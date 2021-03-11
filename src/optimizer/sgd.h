#ifndef __SGD_H
#define __SGD_H

/**
 * implement a optimized SGD
 * */

#include "../optimizer.h"
#include <unordered_map>

class SGD: public Optimizer {
    private:
        double momentum;
        bool nesterov;
        std::unordered_map<const double*, Vector> v_map;

    public:
        explicit SGD(double learning_rate = 0.01, double decay = 0.0,
                     double momentum = 0.0, bool nesterov = false) : Optimizer(learning_rate, decay),
                     momentum(momentum), nesterov(nesterov) {}
        void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw);
};

#endif