//
// Created by jakub-ciesko on 10/6/25.
//
#include "tensor.h"
#include <stdexcept>
#include <iostream>

Tensor::Tensor(const Matrix& value, bool requires_grad)
    : value(value),
      grad(value.rows(), value.cols(), Matrix::InitMethod::ZERO),
      requires_grad(requires_grad),
      backward_fn(nullptr) {}

void Tensor::backward() {
    if (!requires_grad)
        throw std::runtime_error("Cannot call backward() on tensor that does not require grad");
    if (grad.rows() == 1 && grad.cols() == 1)
        grad(0,0) = 1.0f;
    if (backward_fn)
        backward_fn();
}

void Tensor::zero_grad() {
    grad = Matrix(grad.rows(), grad.cols(), Matrix::InitMethod::ZERO);
}


// todo
Tensor Tensor::operator+(const Tensor & other) const {
    Matrix result_value = value + other.value;
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);
    if (result_requires_grad) {
        auto self = *this;
        auto other_copy = other;
        result.backward_fn = [self, other_copy, &result]() mutable {
            if (self.requires_grad)
                self.grad = self.grad + result.grad;
            if (other_copy.requires_grad)
                other_copy.grad = other_copy.grad + result.grad; // propagation of gradients

        };
    }
    return result;
}

Tensor Tensor::operator-(const Tensor & other) const {
    Matrix result_value = value - other.value;
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);
    if (result_requires_grad) {
        auto self = *this;
        auto other_copy = other;
        result.backward_fn = [self, other_copy, &result]() mutable {
            if (self.requires_grad)
                self.grad = self.grad + result.grad;
            if (other_copy.requires_grad)
                other_copy.grad = other_copy.grad - result.grad; // propagation of gradients

        };
    }
    return result;
}

Tensor Tensor::elementwise_multiply(const Tensor & other) const {
    if (value.cols() != other.value.cols() && value.rows() != other.value.rows()) {
        throw std::runtime_error("Tensor::elementwise_multiply() has wrong size");
    }
    Matrix result_value(value.rows(), value.cols());
    for (int i = 0; i < value.rows(); ++i) {
        for (int j = 0; j < value.cols(); ++j) {
            result_value(i, j) = value(i, j)*other.value(i, j); // will delegate to matrix later
        }
    }
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);
    if (result_requires_grad) {
        auto self = *this;
        auto other_copy = other;
        result.backward_fn = [self, other_copy, &result]() mutable {
            if (self.requires_grad)
                self.grad = self.grad + other_copy.value * result.grad; // chain rule
            if (other_copy.requires_grad)
                other_copy.grad = other_copy.grad + self.value * result.grad; // chain rule
        };
    }
    return result;
}
