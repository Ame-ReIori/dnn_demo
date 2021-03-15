#ifndef __UTILS_H
#define __UTILS_H

/**
 * there are some aid functions used in dnn
 * */

#include <eigen3/Eigen/Core>
#include <iostream>
#include <algorithm>
#include <random>
#include <ios>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Array<double, 1, Eigen::Dynamic> RowVector;

static std::random_device seed_gen;
static std::mt19937_64 prg;


// the randomness follows normal distribution
inline void set_normal_random(double *arr, int n, double mu, double sigma) {
    std::normal_distribution<double> dis(mu, sigma);
    for (int i = 0; i < n; i++) {
        arr[i] = dis(prg);
    }
}

inline uint64_t reverse64(uint64_t num) {
    num = ((num & 0xffffffff00000000) >> 32) | ((num & 0x00000000ffffffff) << 32);
    num = ((num & 0xffff0000ffff0000) >> 16) | ((num & 0x0000ffff0000ffff) << 16);
    num = ((num & 0xff00ff00ff00ff00) >> 8 ) | ((num & 0x00ff00ff00ff00ff) << 8 );
    return num;
}

#ifdef DEBUG
    #include <typeinfo>
    #include <iomanip>
#endif

#endif