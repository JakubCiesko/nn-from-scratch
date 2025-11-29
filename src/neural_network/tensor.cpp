//
// Created by jakub-ciesko on 10/6/25.
//
#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <utility>
#include <unordered_set>


Tensor::Tensor(const Matrix &value, bool requires_grad)
    : value(value), grad(value.rows(), value.cols(), Matrix::InitMethod::ZERO),
      requires_grad(requires_grad), backward_fn(nullptr)
{
}

void Tensor::backward()
{
    // seed gradient
    if (grad.rows() == 1 && grad.cols() == 1)
        grad(0,0) = 1.0f;

    // topo order
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;

    std::function<void(Tensor*)> dfs = [&](Tensor* t) {
        if (!t || visited.count(t)) return;
        visited.insert(t);
        for (auto* p : t->parents)
            dfs(p);
        topo.push_back(t);
    };

    dfs(this);

    // reverse pass
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor* t = *it;
        if (t->backward_fn)
            t->backward_fn();
    }
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
        result.parents = { this, &other };
        result.backward_fn = [this, &other, &result]() mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad;
                //this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + result.grad; // propagation of gradients
                //other.backward();
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
        result.parents = { this, &other };
        result.backward_fn = [this, &other, &result]() mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad;
                //this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad - result.grad; // propagation of gradients
                //other.backward();
            }
        };
    }
    return result;
}

Tensor Tensor::elementwise_multiply(Tensor &other)
{
    Matrix result_value = value.elementwise_multiply(other.value);
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);
    if (result_requires_grad)
    {
        result.parents = { this, &other };
        result.backward_fn = [this, &other, &result]() mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + other.value.elementwise_multiply(result.grad);
                //this->backward();
            }

            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.elementwise_multiply(result.grad);
                //other.backward();
            }
        };
    }
    return result;
}

Tensor Tensor::operator*(Tensor &other)
{
    Matrix result_value = value * other.value;
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);

    if (result_requires_grad)
    {
        result.parents = { this, &other };
        result.backward_fn = [this, &other, &result]() mutable
        {
            // C = AB -> grad(A) = grad(C)trans(B), grad(B) = trans(A) grad(C)
            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad * other.value.transpose();
                //this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.transpose() * result.grad;
                //other.backward();
            }
        };
    }
    return result;
}

Tensor Tensor::broadcast_add(Tensor &other, int axis)
{
    Matrix result_value = value.broadcast_add(other.value, axis);
    bool result_requires_grad = requires_grad || other.requires_grad;
    Tensor result(result_value, result_requires_grad);

    if (result_requires_grad)
    {
        result.parents = { this, &other };
        result.backward_fn = [this, &other, &result, axis]() mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad;
                //this->backward();
            }

            if (other.requires_grad)
            {
                other.grad = other.grad + result.grad.sum_over(axis);
                //other.backward();
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
        result.parents = { this };
        result.backward_fn = [this, &result]() mutable
        {
            if (this->requires_grad)
            {
                Matrix mask = this->value.apply([](float val)
                                                { return (val < 0.0f) ? 0.0f : 1.0f; });
                this->grad = this->grad + result.grad.elementwise_multiply(mask);
                //this->backward();
            };
        };
    }
    return result;
}

Tensor Tensor::cross_entropy_loss(const Tensor &y_true)
{
    const Matrix &logits = this->value;
    const Matrix &labels = y_true.value;
    int batch_size = logits.rows();
    int num_classes = logits.cols();

    Matrix max_logits = logits.max_over(1);
    Matrix safe_logits = logits.broadcast_add(max_logits * -1.0f, 1);

    Matrix exps = safe_logits.apply(exp);
    Matrix sum_exps = exps.sum_over(1);

    Matrix log_sum_exps = sum_exps.apply(log);
    Matrix log_softmax = safe_logits.broadcast_add(log_sum_exps * -1.0f, 1);

    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; ++i)
    {
        int correct_class = static_cast<int>(labels.get(i, 0));
        total_loss += log_softmax.get(i, correct_class);
    }
    float mean_loss = -total_loss / static_cast<float>(batch_size);

    Matrix result_value(1, 1, Matrix::InitMethod::ZERO);
    result_value.set(0, 0, mean_loss);
    Tensor result(result_value, true);

    Matrix softmax_probs = exps.broadcast_divide(sum_exps, 1);

    auto softmax_probs_cache = std::move(softmax_probs);
    auto y_true_labels_cache = y_true.value;

    auto &logits_tensor = *this;
    result.parents.push_back(this);
    result.backward_fn = [&logits_tensor, y_true_labels_cache, softmax_probs_cache,
                          num_classes]() mutable
    {
        if (logits_tensor.requires_grad)
        {
            int batch_size = logits_tensor.value.rows();

            Matrix y_true_one_hot(batch_size, num_classes, Matrix::InitMethod::ZERO);
            for (int i = 0; i < batch_size; ++i)
            {
                int correct_class = static_cast<int>(y_true_labels_cache.get(i, 0));
                y_true_one_hot.set(i, correct_class, 1.0f);
            }

            Matrix grad_logits = softmax_probs_cache - y_true_one_hot;
            Matrix mean_grad_logits =
                grad_logits * (1.0f / static_cast<float>(batch_size));

            logits_tensor.grad = logits_tensor.grad + mean_grad_logits;
            //logits_tensor.backward();
        }
    };

    return result;
}

Tensor Tensor::matmul_broadcast_add(Tensor &B, Tensor &C) {
    Matrix result_value = value.matmul_broadcast_add(B.value, C.value);
    Tensor result(result_value, requires_grad || B.requires_grad || C.requires_grad);
    if (result.requires_grad) {
        result.parents = { this, &B, &C };
        result.backward_fn = [this, &B, &C, &result]() mutable
        {

            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad * B.value.transpose();
                //this->backward();
            }
            if (B.requires_grad)
            {
                B.grad = B.grad + this->value.transpose() * result.grad;
                //B.backward();
            }
            if (C.requires_grad) {
                C.grad = C.grad + result.grad.sum_over(0);
                //C.backward();
            }
        };
    }
    return result;
}

void Tensor::matmul(Tensor &other, Tensor &result) {
    value.matmul(other.value, result.value);
    result.requires_grad = other.requires_grad || requires_grad;

    if (result.requires_grad)
    {

        result.backward_fn = [this, &other, &result]() mutable
        {
            // C = AB -> grad(A) = grad(C)trans(B), grad(B) = trans(A) grad(C)
            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad * other.value.transpose();
                //this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.transpose() * result.grad;
                //other.backward();
            }
        };
    }

};
void Tensor::matmul_broadcast_add_prealloc(Tensor &B, Tensor &C, Tensor &result) {
    value.matmul_broadcast_add_prealloc(B.value, C.value, result.value);
    result.requires_grad = requires_grad || B.requires_grad || C.requires_grad ;
    if (result.requires_grad) {
        result.backward_fn = [this, &B, &C, &result]() mutable
        {

            if (this->requires_grad)
            {
                this->grad = this->grad + result.grad * B.value.transpose();
                //this->backward();
            }
            if (B.requires_grad)
            {
                B.grad = B.grad + this->value.transpose() * result.grad;
                //B.backward();
            }
            if (C.requires_grad) {
                C.grad = C.grad + result.grad.sum_over(0);
                //C.backward();
            }
        };
    }
};

void Tensor::relu_inplace() {
    value.apply_inplace([](float x) { return std::max(0.0f, x); });
}

void Tensor::relu_prealloc(Tensor &result) {
    std::function relu_fn = [](float val) { return std::max(0.0f, val); };
    result.value = value.apply(relu_fn);
    result.requires_grad = requires_grad;
    if (result.requires_grad)
    {

        result.backward_fn = [this, &result]() mutable
        {
            if (this->requires_grad)
            {
                Matrix mask = this->value.apply([](float val)
                                                { return (val < 0.0f) ? 0.0f : 1.0f; });
                this->grad = this->grad + result.grad.elementwise_multiply(mask);
                //this->backward();
            };
        };
    }
}