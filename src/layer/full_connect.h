#ifndef __FULL_CONNECT_H
#define __FULL_CONNECT_H

/**
 * implement fully connected layer
 * */

#include "../layer.h"
#include <vector>

class FullConnect: public Layer {
    private:
        const int dim_in;
        const int dim_out;

        // name makes sense
        Matrix weight;
        Vector bias;
        Matrix grad_weight;
        Vector grad_bias;

        void init();
    
    public:
        FullConnect(const int dim_in, const int dim_out) :
                    dim_in(dim_in), dim_out(dim_out) {
                        init();
                    }
        
        void forward(const Matrix& bottom); // due to the existance of layer. the return type is void
        void backward(const Matrix& bottom, const Matrix& grad_top);
        void update(Optimizer& opt);
        int output_dim() { return dim_out; }
        std::vector<double> get_parameters() const;
        std::vector<double> get_derivatives() const; // get gradient
        void set_parameters(const std::vector<double> parameters);
};


#endif