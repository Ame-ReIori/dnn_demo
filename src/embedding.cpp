#include "./embedding.h"
#include "utils.h"
#include <algorithm>
#include <iterator>
#include <vector>

void Embedding::init() {
    std::vector<int> index(size);
    std::vector<double> embedding(size);

    // initial index
    for (int i = 0; i < size; i++) index[i] = i;

    // initial embedding data
    set_normal_random(embedding.data(), size, 0, 0.01);

    auto zip0 = [](const double& d, int index) { return std::make_pair(index, d); }; 
    auto zip1 = [](const double& d, int index) { return std::make_pair(d, index); }; 

    std::transform(embedding.begin(), embedding.end(), 
                   index.begin(), std::inserter(data_to_embed, data_to_embed.end()), zip0);
    std::transform(embedding.begin(), embedding.end(), 
                   index.begin(), std::inserter(embed_to_data, embed_to_data.end()), zip1);
}

void Embedding::embed(Matrix &data, Matrix &embed_data) {
    int rows = data.rows();
    int cols = data.cols();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            embed_data(i, j) = data_to_embed[data(i, j)];
        }
    }
}

void Embedding::inv_embed(Matrix &embed_data, Matrix &data) {
    int rows = embed_data.rows();
    int cols = embed_data.cols();

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data(i, j) = embed_to_data[embed_data(i, j)];
        }
    }
}