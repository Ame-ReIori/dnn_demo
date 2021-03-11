#ifndef __LOSS_H
#define __LOSS_H

/**
 * basic loss function class
 * */


#include "./utils.h"

class Loss {
    protected:
        double loss; // loss value
        Matrix grad; // gradient

    public:
        virtual ~Loss() {}

        virtual void evaluate(const Matrix& pred, const Matrix& target) = 0;
        virtual double output() { return loss; }
        virtual const Matrix& back_gradient() { return grad; }
};

#endif