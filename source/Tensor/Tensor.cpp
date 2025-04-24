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

    Tensor& Tensor::operator+=(const Tensor &other) {
        Tensor broadcastedOther = other.broadcastTo(*this);
        if (this->shape != broadcastedOther.shape)
            throw std::runtime_error("Broadcasting failed, shapes do not match after broadcast.");
        for (size_t i = 0; i < this->size; ++i)
            this->data[i] += broadcastedOther.data[i];
        return *this;
    }

    Tensor Tensor::operator-=(const Tensor &other) {
        Tensor broadcastedOther = other.broadcastTo(*this);
        if (this->shape != broadcastedOther.shape)
            throw std::runtime_error("Broadcasting failed, shapes do not match after broadcast.");
        for (size_t i = 0; i < this->size; ++i)
            this->data[i] -= broadcastedOther.data[i];
        return *this;
    }

    Tensor Tensor::operator*=(const Tensor &other){

        return *this;
    }

    Tensor Tensor::broadcastTo(const Tensor& target) const {
        std::vector<size_t> broadcastedShape = getBroadcastedShape(this->shape, target.shape);
        Tensor result(broadcastedShape);

        std::vector<size_t> extendedShape = this->shape;
        if (extendedShape.size() < broadcastedShape.size()) {
            extendedShape.insert(
                    extendedShape.begin(),
                    broadcastedShape.size() - extendedShape.size(),
                    1
            );
        }

        for (size_t i = 0; i < result.size; i++) {
            auto multiIndex = unravel(i, broadcastedShape);
            auto sourceMultiIndex = multiIndex;
            for (size_t axis = 0; axis < sourceMultiIndex.size(); axis++) {
                if (extendedShape[axis] == 1)
                    sourceMultiIndex[axis] = 0;
            }

            size_t sourceFlatIndex = ravel(sourceMultiIndex, extendedShape);
            result.data[i] = this->data[sourceFlatIndex];
        }

        return result;
    }

    std::vector<size_t> Tensor::getBroadcastedShape(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2) {
        size_t maxLen = std::max(shape1.size(), shape2.size());
        std::vector<size_t> s1 = shape1, s2 = shape2;

        if (s1.size() < maxLen) s1.insert(s1.begin(), maxLen - s1.size(), 1);
        if (s2.size() < maxLen) s2.insert(s2.begin(), maxLen - s2.size(), 1);

        std::vector<size_t> resultShape(maxLen);
        for (size_t i = 0; i < maxLen; ++i) {
            if (s1[i] == s2[i] or s2[i] == 1)
                resultShape[i] = s1[i];
            else if (s1[i] == 1)
                resultShape[i] = s2[i];
            else
                throw std::invalid_argument("Shapes are not broadcastable");
        }

        return resultShape;
    }

    std::vector<size_t> Tensor::unravel(size_t index, const std::vector<size_t> & shape) {
        std::vector<size_t> multiIndex(shape.size());
        for (size_t i = 0; i < shape.size(); i++){
            size_t stride = 1;
            for (size_t j = i + 1; j < shape.size(); j++)
                stride *= shape[j];
            multiIndex[i] = (index / stride) % shape[i];
        }
        return multiIndex;
    }


    size_t Tensor::ravel(const std::vector<size_t>& multiIndex, const std::vector<size_t>& shape) {
        size_t index = 0;
        size_t stride = 1;
        for (size_t i = shape.size(); i-- > 0;){
            index += multiIndex[i] * stride;
            stride *= shape[i];
        }
        return index;
    }


    Tensor operator+(const Tensor& a, const Tensor& b) {
        auto result = Tensor(Tensor::getBroadcastedShape(a.shape, b.shape));
        auto tensorA = a.broadcastTo(result);
        auto tensorB = b.broadcastTo(result);
        for (size_t i = 0; i < result.size; ++i)
            result.data[i] = tensorA.data[i] + tensorB.data[i];
        return result;
    }



    Tensor operator-(const Tensor& a, const Tensor& b){
        auto result = Tensor(Tensor::getBroadcastedShape(a.shape, b.shape));
        auto tensorA = a.broadcastTo(result);
        auto tensorB = b.broadcastTo(result);
        for (size_t i = 0; i < result.size; ++i)
            result.data[i] = tensorA.data[i] - tensorB.data[i];
        return result;
    }
}