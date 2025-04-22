#include <iostream>
#include <vector>
#include "Tensor/Tensor.h"
#include "Tensor/Initializers.h"


int main() {
    using namespace Tensors;

    // Примеры размеров
    std::vector<size_t> shape1 = {5, 1, 4};
    std::vector<size_t> shape2 = {1, 3, 4};

    auto tensor1 = Tensor(shape1);
    auto tensor2 = Tensor(shape2);

    try {
        // Вызываем нашу функцию
        Tensor result = Tensor::broadcast(tensor1, tensor2);

        // Печатаем результат
        std::cout << "Broadcasted shape: (";
        for (size_t i = 0; i < result.shape.size(); ++i) {
            std::cout << result.shape[i];
            if (i != result.shape.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }
    catch (const std::invalid_argument& e) {
        std::cerr << "Broadcast failed: " << e.what() << std::endl;
    }

    return 0;
}


