#ifndef CONVOLUTIONALNN_INITIALIZERS_H
#define CONVOLUTIONALNN_INITIALIZERS_H

#include <vector>
#include <random>
#include <cmath>

namespace Initializers{
    void he(float* , const std::vector<size_t>&, size_t);
    void ones(float*, size_t);
}

#endif //CONVOLUTIONALNN_INITIALIZERS_H
