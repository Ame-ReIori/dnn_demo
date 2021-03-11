#ifndef __OPTIMIZER_H
#define __OPTIMIZER_H

/**
 * basic optimizer class
 * */

#include "./utils.h"

class Optimizer {
    protected:
        double learning_rate;
        double decay; // changes adaptively according to learning rate

    public:
        explicit Optimizer(double learning_rate = 0.01, double decay = 0.0): // avoid implicit conversion
            learning_rate(learning_rate), decay(decay) {}
        virtual ~Optimizer() {} // to be defined

        /** 
         * para @w: weights
         * pare @dw: gradients
         * */
        virtual void update(Vector::AlignedMapType& w, Vector::ConstAlignedMapType& dw) = 0;
};

#endif