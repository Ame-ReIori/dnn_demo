#ifndef __DATA_H
#define __DATA_H

/**
 * process data
 * */

#include "./utils.h"
#include <string>

class Data {
    private:
        std::string data_dir;

    public:
        Matrix train_data;
        Matrix train_label;
        Matrix test_data;
        Matrix test_label;

        explicit Data(std::string dir) : data_dir(dir) {}

        void read(std::string filename, int feature_num, 
                    int begin, int end, Matrix& data);
};


#endif