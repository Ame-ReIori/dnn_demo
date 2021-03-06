#include "./data.h"
#include "./utils.h"
#include <fstream>

void Data::read(std::string filename, int feature_num, 
                    int begin, int end, Matrix &data) {
    double tmp;
    int sample_num = end - begin;

    data.resize(feature_num, sample_num);
    std::ifstream inFile(data_dir + filename, std::ios::in|std::ios::binary);
    for (int i = 0; i < sample_num; i++) {
        for (int j = 0; j < feature_num; j++) {
            inFile.read((char *)&tmp, 8);
            data(j, i) = tmp;
        }
    }
}