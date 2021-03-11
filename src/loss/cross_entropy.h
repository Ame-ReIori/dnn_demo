#ifndef __CROSS_ENTROPY_H
#define __CROSS_ENTROPY_H

/**
 * implement cross entropy loss function
 * */

#include "../loss.h"

class CrossEntropy: public Loss {
    public:
        void evaluate(const Matrix& pred, const Matrix& target);
};

#endif