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

        friend Tensor operator+(const Tensor&, const Tensor&);
        friend Tensor operator-(const Tensor&, const Tensor&);

        explicit Tensor(const std::vector<size_t> &);
        Tensor(const std::vector<size_t> &, const float*);
        Tensor(const std::vector<size_t> &, Initilizer);
        Tensor(const Tensor&);
        Tensor(Tensor&&) noexcept;

        ~Tensor();

        float& operator[](size_t);
        const float& operator[](size_t) const;
        Tensor& operator+=(const Tensor&);
        Tensor operator-=(const Tensor&);
        Tensor operator*=(const Tensor&);
        Tensor operator*(const float) const;

        Tensor broadcastTo( const Tensor&) const;

    private:
        void allocate();
        void copyData(const float*);
        static std::vector<size_t> unravel(size_t, const std::vector<size_t>&);
        static size_t ravel(const std::vector<size_t>&, const std::vector<size_t>&);
        static std::vector<size_t> getBroadcastedShape(const std::vector<size_t>&, const std::vector<size_t>&);
    };

    Tensor operator+(const Tensor&, const Tensor&);
    Tensor operator-(const Tensor&, const Tensor&);


}
#endif //UNTITLED_TENSOR_H
