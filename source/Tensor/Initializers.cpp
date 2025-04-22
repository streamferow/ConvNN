#include "Tensor/Initializers.h"

namespace Initializers{

    void he(float* data, const std::vector<size_t>& shape, size_t size){
        size_t fanIn = 1;
        float gain = 1.0;

        if(shape.size() > 1) fanIn = std::accumulate(shape.begin() + 1, shape.end(), size_t(1), std::multiplies<size_t>());
        else fanIn = shape[0];

        const float stddev = gain * std::sqrt(2.0 / static_cast<float>(fanIn));

        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<float> distribution(0.0, stddev);

        for (size_t i = 0; i < size; i++) data[i] = distribution(gen);
    }

}

