#ifndef __UTILS_H
#define __UTILS_H

/**
 * there are some aid functions used in dnn
 * */

#include <eigen3/Eigen/Core>
#include <iostream>
#include <algorithm>
#include <random>

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic> RowVector;

static std::random_device seed_gen;
static std::mt19937_64 prg;


// the randomness follows normal distribution
inline void set_normal_random(double *arr, int n, double mu, double sigma) {
    std::normal_distribution<double> dis(mu, sigma);
    for (int i = 0; i < n; i++) {
        arr[i] = dis(prg);
    }
}

#ifdef DEBUG
#include <typeinfo>
#endif

#endif