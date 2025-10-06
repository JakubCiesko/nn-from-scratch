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
Tensor Tensor::operator+(const Tensor &) const {
    Tensor result(*this);
    return result;
}
