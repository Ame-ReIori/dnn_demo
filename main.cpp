#include "src/data.h"
#include "src/utils.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/layer.h"
#include "src/layer/full_connect.h"
#include "src/layer/relu.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse.h"
#include "src/loss/cross_entropy.h"
#include <algorithm>
#include <vector>

void p(std::vector<double> vec) {
    std::vector<double>::iterator ite = vec.begin();
    for (; ite != vec.end(); ite++) {
        std::cout << *ite << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    Data d("/home/ame/GraduationProject/Code/demo/dnn_demo/data/");
    
    d.read("features.bin", 2, 0, 200, d.train_data);
    d.read("label.bin", 1, 0, 200, d.train_label);

    #ifdef DEBUG
        // std::cout << std::fixed << std::setprecision(15) << d.train_data << std::endl;
        // std::cout << std::fixed << std::setprecision(15) << d.train_label << std::endl;

        int sample_num = d.train_data.cols();
        int feature_num = d.train_data.rows();

        std::cout << "The number of samples is " << sample_num << std::endl;
        std::cout << "The number of feature is " << feature_num << std::endl;
    #endif

    // construct a nn
    Network dnn;
    FullConnect* fc1 = new FullConnect(2, 10);
    Layer* relu = new ReLU;
    FullConnect* fc2 = new FullConnect(10, 1);
    Layer* softmax = new Softmax;
    
    dnn.add_layer(fc1);
    dnn.add_layer(relu);
    dnn.add_layer(fc2);
    dnn.add_layer(softmax);

    Loss* loss = new CrossEntropy;
    dnn.add_loss(loss);
    SGD opt(0.001, 0.0, 0.0, false);

    const int epoch = 100;
    
    
    for (int i = 0; i < epoch; i++) {
        dnn.forward(d.train_data);
        dnn.backward(d.train_data, d.train_label);

        #ifdef DEBUG  
            std::cout << "the parameters as followed:" << std::endl;
            std::vector<std::vector<double>> para = dnn.get_parameters();
            std::for_each(para.begin(), para.end(), p);
        #endif
        #ifndef DEBUG
            std::cout << i << "-th epoch, loss: " << dnn.get_loss() << std::endl;
        #else
            std::cout << i << "-th epoch, loss: " << std::fixed << std::setprecision(15) << dnn.get_loss() << std::endl;
        #endif

        dnn.update(opt);
    }
    return 0;
}