#include "Tensor/Tensor.h"

namespace Tensors {

    void Tensor::allocate() {
        if (size > 0)
            data = new float[size]();
        else
            data = nullptr;
    }


    void Tensor::copyData(const float *source) {
        if (source and data){
            for (size_t i = 0; i < size; i++)
                data[i] = source[i];
        }
    }


    Tensor::Tensor(const std::vector<size_t> &newShape)
    : data(nullptr),
      shape(newShape),
      size(1)
    {
        for (auto dimension : shape) size *= dimension;
        allocate();
    }


    Tensor::Tensor(const std::vector<size_t> &newShape, const float* newData)
    : data(nullptr),
      shape(newShape),
      size(1)
    {
        for (auto dimension : shape) size *= dimension;
        allocate();
        copyData(newData);
    }


    Tensor::Tensor(const std::vector<size_t>& newShape, Initilizer initializer)
    : shape(newShape),
      data(nullptr),
      size(1)
    {
        for (auto dimension : shape) size *= dimension;
        allocate();
        if (initializer) initializer(data, shape, size);
    }


    Tensor::Tensor(const Tensor& other)
    : data(nullptr),
      shape(other.shape),
      size(other.size)
    {
        allocate();
        copyData(other.data);
    }


    Tensor::Tensor(Tensor&& other) noexcept
    : data(other.data),
      shape(std::move(other.shape)),
      size(other.size)
    {
        other.data = nullptr;
        other.size = 0;
    }


    Tensor::~Tensor() {
        delete[] data;
    }


    float & Tensor::operator[](size_t index) {
        assert(index < size);
        return data[index];
    }


    const float & Tensor::operator[](size_t index) const {
        assert(index < size);
        return data[index];
    }


    Tensor Tensor::operator+(const Tensor &other) const {
        assert(shape == other.shape);
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++) result[i] = this->data[i] + other.data[i];
        return result;
    }


    Tensor Tensor::operator-(const Tensor &other) const {
        assert(shape == other.shape);
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++) result[i] = this->data[i] - other.data[i];
        return result;
    }

    Tensor Tensor::operator*(const float scalar) const {
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++) result[i] = this->data[i] * scalar;
        return result;
    }


    Tensor Tensor::operator*(const Tensor &other) const {
        assert(shape == other.shape);
        Tensor result(this->shape);
        for (size_t i = 0; i < size; i++) result[i] = this->data[i] * other.data[i];
        return result;
    }


    Tensor Tensor::broadcast(Tensor tensor1, Tensor tensor2){
        auto s1 = tensor1.shape;
        auto s2 = tensor2.shape;

        size_t maxLen = std::max(s1.size(), s2.size());

        if (s1.size() < maxLen) s1.insert(s1.begin(), maxLen - s1.size(), 1);
        if (s2.size() < maxLen) s2.insert(s2.begin(), maxLen - s2.size(), 1);

        std::vector<size_t> broadcastedShape(maxLen);

        for (size_t i = 0; i < maxLen; i++){
            if (s1[i] == s2[i])
                broadcastedShape[i] = s1[i];
            else if (s1[i] == 1)
                broadcastedShape[i] = s2[i];
            else if (s2[i] == 1)
                broadcastedShape[i] = s1[i];
            else
                throw std::invalid_argument("Shapes are not broadcastable");
        }
        return Tensor(broadcastedShape);
    }


}