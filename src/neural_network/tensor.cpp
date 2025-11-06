//
// Created by jakub-ciesko on 10/6/25.
//
#include "tensor.h"
#include <cmath>
#include <stdexcept>

Tensor::Tensor(const Matrix &value, bool requires_grad)
    : value(value), grad(value.rows(), value.cols(), Matrix::InitMethod::ZERO),
      requires_grad(requires_grad), backward_fn(nullptr)
{
}

void Tensor::backward()
{
    if (!requires_grad)
        throw std::runtime_error(
            "Cannot call backward() on tensor that does not require grad");
    if (grad.rows() == 1 && grad.cols() == 1)
        grad(0, 0) = 1.0f;
    if (backward_fn)
        backward_fn();
}

void Tensor::zero_grad()
{
    grad = Matrix(grad.rows(), grad.cols(), Matrix::InitMethod::ZERO);
}

// todo
Tensor Tensor::operator+(Tensor &other)
{
    Matrix result_value = value + other.value;
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);
    if (result_requires_grad)
    {
        auto self = *this;
        auto other_copy = other;
        result.backward_fn = [this, &other, &result]() mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad;
                this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + result.grad; // propagation of gradients
                other.backward();
            }
        };
    }
    return result;
}

Tensor Tensor::operator-(Tensor &other)
{
    Matrix result_value = value - other.value;
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);
    if (result_requires_grad)
    {
        auto self = *this;
        auto other_copy = other;
        result.backward_fn = [this, &other, &result]() mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad;
                this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad - result.grad; // propagation of gradients
                other.backward();
            }
        };
    }
    return result;
}

Tensor Tensor::elementwise_multiply(Tensor &other)
{
    if (value.cols() != other.value.cols() && value.rows() != other.value.rows())
    {
        throw std::runtime_error("Tensor::elementwise_multiply() has wrong size");
    }
    Matrix result_value = value.elementwise_multiply(other.value);
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);
    if (result_requires_grad)
    {
        auto self = *this;
        auto other_copy = other;
        result.backward_fn = [this, &other, &result]() mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + other.value.elementwise_multiply(result.grad);
                this->backward();
            } // chain rule

            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.elementwise_multiply(result.grad);
                other.backward();
            } // chain rule
        };
    }
    return result;
}

Tensor Tensor::operator*(Tensor &other)
{
    if (value.cols() != other.value.rows())
    {
        throw std::runtime_error("Tensor::operator*(): Tensors have wrong size (" +
                                 std::to_string(value.rows()) + ", " +
                                 std::to_string(value.cols()) + ") x (" +
                                 std::to_string(other.value.rows()) + "," +
                                 std::to_string(other.value.cols()) + ")");
    }
    Matrix result_value = value * other.value;
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);

    if (result_requires_grad)
    {
        auto self = *this;
        auto other_copy = other;
        result.backward_fn = [this, &other, &result]() mutable
        {
            // C = AB -> grad(A) = grad(C)trans(B), grad(B) = trans(A) grad(C)
            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad * other.value.transpose();
                this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.transpose() * result.grad;
                other.backward();
            }
        };
    }
    return result;
}

Tensor Tensor::broadcast_add(Tensor &other, int axis)
{
    // i let matrix class handle the errors
    Matrix result_value = value.broadcast_add(other.value, axis);
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);

    if (result_requires_grad)
    {
        auto self = *this;
        auto other_copy = other;
        result.backward_fn = [this, &other, &result, axis]() mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad;
                this->backward();
            }

            if (other.requires_grad)
            {
                other.grad = other.grad + result.grad.sum_over(axis);
                other.backward();
            }
        };
    }
    return result;
}

Tensor Tensor::relu()
{
    std::function relu_fn = [](float val) { return std::max(0.0f, val); };
    Matrix result_value = value;
    result_value.apply_inplace(relu_fn);
    Tensor result(result_value, requires_grad);
    if (result.requires_grad)
    {
        auto self = *this;
        result.backward_fn = [this, &result]() mutable
        {
            if (this->requires_grad)
            {
                Matrix mask = this->value.apply([](float val)
                                                { return (val < 0.0f) ? 0.0f : 1.0f; });
                this->grad = this->grad + result.grad.elementwise_multiply(mask);
                this->backward();
            };
        };
    }
    return result;
}
