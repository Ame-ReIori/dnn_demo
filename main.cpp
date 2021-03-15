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
    
    d.read("feature.bin", 2, 0, 180, d.train_data);
    d.read("label.bin", 1, 0, 180, d.train_label);
    d.read("feature.bin", 2, 181, 200, d.test_data);
    d.read("label.bin", 1, 181, 200, d.test_label);

    #ifdef DEBUG
        std::cout << std::fixed << std::setprecision(15) << d.train_data << std::endl;
        std::cout << std::fixed << std::setprecision(15) << d.train_label << std::endl;

        int sample_num = d.train_data.cols();
        int feature_num = d.train_data.rows();

        std::cout << "The number of samples is " << sample_num << std::endl;
        std::cout << "The number of feature is " << feature_num << std::endl;
    #endif

    // construct a nn
    Network dnn;
    FullConnect* fc1 = new FullConnect(2, 10);
    Layer* relu = new ReLU;
    FullConnect* fc2 = new FullConnect(10, 2);
    Layer* softmax = new Softmax;
    
    dnn.add_layer(fc1);
    dnn.add_layer(relu);
    dnn.add_layer(fc2);
    dnn.add_layer(softmax);

    Loss* loss = new CrossEntropy;
    dnn.add_loss(loss);
    SGD opt(0.001, 0.0, 0.0, false);


    #ifdef DEBUG 
        // set weight
        double tmp0[30] = {-0.0098, -0.6372, 0.5496,  0.2175, -0.1499,  0.3334, -0.4066,  0.4858, -0.2002,  0.5076, 0.3578, -0.1654, 0.2966, -0.5829, -0.0709, -0.3585, 0.6268, -0.2705, -0.3856, -0.3769, -0.5413, -0.4499,  0.6506, -0.1143, -0.4466, -0.6392, -0.6528,  0.6737, -0.5349,  0.5171};
        double tmp1[22] = {-0.1123,  0.0187,  0.1242, -0.0546, -0.2224, -0.1411, -0.1274, -0.2274, -0.0764, -0.2944, -0.1347,  0.0678, -0.2769, -0.1391, -0.0407, -0.0529,  0.1596, -0.2312, 0.1370,  0.1135, -0.1202, -0.1526};
        
        std::vector<double> first_linear(tmp0, tmp0+30);
        std::vector<double> second_linear(tmp1, tmp1+22);

        fc1->set_parameters(first_linear);
        fc2->set_parameters(second_linear);

        std::cout << "the parameters as followed:" << std::endl;
        std::vector<std::vector<double>> para = dnn.get_parameters();
        std::for_each(para.begin(), para.end(), p);
    #endif

    const int epoch = 100;
    
    
    for (int i = 0; i < epoch; i++) {
        dnn.forward(d.train_data);
        dnn.backward(d.train_data, d.train_label);

        #ifndef DEBUG
            std::cout << i << "-th epoch, loss: " << dnn.get_loss() << std::endl;
        #else
            std::cout << i << "-th epoch, loss: " << std::fixed << std::setprecision(15) << dnn.get_loss() << std::endl;
        #endif

        dnn.update(opt);
    }
    dnn.forward(d.test_data);
    double acc = compute_accuracy(dnn.output(), d.test_label);
    std::cout << acc << std::endl;
    return 0;
}