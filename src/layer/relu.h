#ifndef __RELU_H
#define __RELU_H

/**
 * implement ReLU activiate function
 * */

#include "../layer.h"

class ReLU: public Layer {
    public:
        void forward(const Matrix& bottom);
        void backward(const Matrix& bottom, const Matrix& grad_top);
};

#endif