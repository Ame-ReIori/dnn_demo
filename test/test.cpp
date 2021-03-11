#include "../utils.h"
#include <algorithm>

void p(double a) {
    std::cout << a << ' ';
}

int main() {
    double arr[10];
    set_normal_random(arr, 10, 0.1, 0.01);
    std::for_each(arr, arr+9, p);
    std::cout << std::endl;
    std::cout << arr << std::endl;
}