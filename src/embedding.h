#ifndef __EMBEDDING_H
#define __EMBEDDING_H

/**
 * this is the class to embed data.
 * embed tabel is stored by recommendation server
 * client should communicate with server when needing to embed data
 * */

#include "./utils.h"
#include <unordered_map>

class Embedding {
    private:
        const int size;
        void init();

    public:
        std::unordered_map<int, double> data_to_embed;
        std::unordered_map<double, int> embed_to_data;

        Embedding(const int size) : size(size) {
            init();
        }

        void embed(Matrix& data, Matrix& embed_data);
        void inv_embed(Matrix& embed_data, Matrix& data);
};

#endif