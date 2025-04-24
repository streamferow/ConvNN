#include <iostream>
#include <vector>
#include "Tensor/Tensor.h"
#include "Tensor/Initializers.h"


int main() {
    using namespace Tensors;

    // Примеры размеров
    std::vector<size_t> shape1 = {5, 3, 4};
    std::vector<size_t> shape2 = {1, 3, 4};

    auto tensor1 = Tensor(shape1);
    auto tensor2 = Tensor(shape2);

    auto res = tensor1 + tensor2;

    return 0;
}



