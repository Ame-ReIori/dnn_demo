#ifndef __SOFTMAX_H
#define __SOFTMAX_H

/**
 * implement softmax activiate function
 * */

#include "../layer.h"

class Softmax: public Layer {
    public:
        void forward(const Matrix& bottom);
        void backward(const Matrix& bottom, const Matrix& grad_top);
};

#endif