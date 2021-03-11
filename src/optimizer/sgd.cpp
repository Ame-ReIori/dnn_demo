#include "./sgd.h"

void SGD::update(Vector::AlignedMapType &w, Vector::ConstAlignedMapType &dw) {
    Vector v = v_map[dw.data()];
    
    if (v.size() == 0) {
        v.resize(dw.size());
        v.setZero();
    }

    v = momentum * v + dw + decay * w;

    if (nesterov) {
        w -= learning_rate * (momentum * v + dw + decay * w);
    } else {
        w -= learning_rate * v;
    }
}