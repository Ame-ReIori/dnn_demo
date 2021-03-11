#ifndef __MSE_H
#define __MSE_H

/**
 * implement mse loss function
 * */

#include "../loss.h"

class MSE: public Loss {
    public:
        void evaluate(const Matrix& pred, const Matrix& target);
};

#endif