#ifndef UNTITLED_TENSOR_H
#define UNTITLED_TENSOR_H

#include <vector>
#include <string>
#include <functional>
#include <iostream>
#include <cassert>
#include "Initializers.h"

namespace Tensors {

    class Tensor {
    public:
        float* data;
        std::vector<size_t> shape;
        size_t size;

        using Initilizer = std::function<void(float*, const std::vector<size_t> &, size_t)>;

        explicit Tensor(const std::vector<size_t> &);
        Tensor(const std::vector<size_t> &, const float*);
        Tensor(const std::vector<size_t> &, Initilizer);
        Tensor(const Tensor&);
        Tensor(Tensor&&) noexcept;

        ~Tensor();

        float& operator[](size_t);
        const float& operator[](size_t) const;
        Tensor operator+(const Tensor&) const;
        Tensor operator-(const Tensor&) const;
        Tensor operator*(const Tensor&) const;
        Tensor operator*(const float) const;

        static Tensor broadcast( Tensor, Tensor);

    private:
        void allocate();
        void copyData(const float*);
    };

}
#endif //UNTITLED_TENSOR_H
