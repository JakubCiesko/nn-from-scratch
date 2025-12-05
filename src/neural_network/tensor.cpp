//
// Created by jakub-ciesko on 10/6/25.
//
#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <utility>
#include <unordered_set>



// Tensor is emballage for Matrix + Gradients.
// TODO: Implement individual layers as subclasses of Tensor
Tensor::Tensor(const Matrix &value, bool requires_grad)
    : value(value), grad(value.rows(), value.cols(), Matrix::InitMethod::ZERO),
      requires_grad(requires_grad), backward_fn(nullptr){};

void Tensor::zero_grad()
{
    grad = Matrix(grad.rows(), grad.cols(), Matrix::InitMethod::ZERO);
}

void Tensor::backward()
{
    // seed gradient
    if (grad.rows() == 1 && grad.cols() == 1)
        grad(0,0) = 1.0f;

    // topological order to visit all parents
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
        if (Tensor* t = *it; t->backward_fn)
            t->backward_fn(*t);
    }
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
        result.backward_fn = [this, &other](const Tensor& self) mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + self.grad;
                //this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + self.grad; // propagation of gradients
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
        result.backward_fn = [this, &other](const Tensor& self) mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + self.grad;
                //this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad - self.grad; // propagation of gradients
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
        result.backward_fn = [this, &other](const Tensor &self) mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + other.value.elementwise_multiply(self.grad);
                //this->backward();
            }

            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.elementwise_multiply(self.grad);
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
        result.backward_fn = [this, &other](const Tensor &self) mutable
        {
            // C = AB -> grad(A) = grad(C)trans(B), grad(B) = trans(A) grad(C)
            if (this->requires_grad)
            {
                this->grad = this->grad + self.grad * other.value.transpose();
                //this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.transpose() * self.grad;
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
        result.backward_fn = [this, &other, axis](const Tensor &self) mutable
        {
            if (this->requires_grad)
            {
                this->grad = this->grad + self.grad;
                //this->backward();
            }

            if (other.requires_grad)
            {
                other.grad = other.grad + self.grad.sum_over(axis);
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
        result.backward_fn = [this](const Tensor &self) mutable
        {
            if (this->requires_grad)
            {
                Matrix mask = this->value.apply([](float val)
                                                { return (val < 0.0f) ? 0.0f : 1.0f; });
                this->grad = this->grad + self.grad.elementwise_multiply(mask);
                //this->backward();
            };
        };
    }
    return result;
}

/*
 * function to stabilize logit values -- shifts them row-wise by row maximum
 * to prevent overflow when calculating exponential
 */
Matrix stabilize_logits(const Matrix &logits)
{
    Matrix max_logits = logits.max_over(1);
    return logits.broadcast_add(max_logits * -1.0f, 1);
}


std::tuple<Matrix, Matrix, Matrix> log_softmax_rows(const Matrix &safe_logits)
{
    Matrix exps = safe_logits.apply(exp);      // exp(z)
    Matrix sum_exps = exps.sum_over(1);        // Σ exp(z)
    Matrix log_sum_exps = sum_exps.apply(log); // log(Σ exp(z))

    // log softmax = z - log_sum_exps
    Matrix log_softmax =
        safe_logits.broadcast_add(log_sum_exps * -1.0f, 1);

    return {log_softmax, exps, sum_exps};
}

Tensor Tensor::cross_entropy_loss(const Tensor &y_true)
{
    const Matrix &logits = this->value;
    const Matrix &labels = y_true.value;
    int batch_size = logits.rows();
    int num_classes = logits.cols();


    Matrix safe_logits = stabilize_logits(logits);

    //Compute log-softmax, exp(safe_logits), and their row sums
    auto result_tuple = log_softmax_rows(safe_logits);
    Matrix log_softmax = std::get<0>(result_tuple);
    Matrix exps        = std::get<1>(result_tuple);
    Matrix sum_exps    = std::get<2>(result_tuple);


    float total_loss = 0.0f;
    for (int i = 0; i < batch_size; ++i)
    {
        int correct_class = static_cast<int>(labels.get(i, 0));
        total_loss += log_softmax.get(i, correct_class);
    }
    // average negative log likelihood
    float mean_loss = -total_loss / static_cast<float>(batch_size);

    Matrix result_value(1, 1, Matrix::InitMethod::ZERO);
    result_value.set(0, 0, mean_loss);
    Tensor result(result_value, true);
    // softmax = exp / sum(exp)
    // softmax values for each class
    Matrix softmax_probs = exps.broadcast_divide(sum_exps, 1);

    auto softmax_probs_cache = std::move(softmax_probs);
    auto y_true_labels_cache = y_true.value;

    auto &logits_tensor = *this;
    result.parents.push_back(this);
    result.backward_fn = [&logits_tensor, y_true_labels_cache, softmax_probs_cache,
                          num_classes](const Tensor &self) mutable
    {

        if (logits_tensor.requires_grad)
        {
            const int batch_size = logits_tensor.value.rows();

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
        result.backward_fn = [this, &B, &C](const Tensor &self) mutable
        {

            if (this->requires_grad)
            {
                this->grad = this->grad + self.grad * B.value.transpose();
                //this->backward();
            }
            if (B.requires_grad)
            {
                B.grad = B.grad + this->value.transpose() * self.grad;
                //B.backward();
            }
            if (C.requires_grad) {
                C.grad = C.grad + self.grad.sum_over(0);
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

        result.backward_fn = [this, &other](const Tensor &self) mutable
        {
            // C = AB -> grad(A) = grad(C)trans(B), grad(B) = trans(A) grad(C)
            if (this->requires_grad)
            {
                this->grad = this->grad + self.grad * other.value.transpose();
                //this->backward();
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.transpose() * self.grad;
                //other.backward();
            }
        };
    }

};
void Tensor::matmul_broadcast_add_prealloc(Tensor &B, Tensor &C, Tensor &result) {
    value.matmul_broadcast_add_prealloc(B.value, C.value, result.value);
    result.requires_grad = requires_grad || B.requires_grad || C.requires_grad ;
    if (result.requires_grad) {
        result.backward_fn = [this, &B, &C](const Tensor &self) mutable
        {

            if (this->requires_grad)
            {
                this->grad = this->grad + self.grad * B.value.transpose();
                //this->backward();
            }
            if (B.requires_grad)
            {
                B.grad = B.grad + this->value.transpose() * self.grad;
                //B.backward();
            }
            if (C.requires_grad) {
                C.grad = C.grad + self.grad.sum_over(0);
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

        result.backward_fn = [this](const Tensor &self) mutable
        {
            if (this->requires_grad)
            {
                Matrix mask = this->value.apply([](float val)
                                                { return (val < 0.0f) ? 0.0f : 1.0f; });
                this->grad = this->grad + self.grad.elementwise_multiply(mask);
                //this->backward();
            };
        };
    }
}


Tensor Tensor::dropout(const float p, const bool training=true) {
    Tensor result = *this;
    // dropout is not to be used in test pass, also drop invalid probability values
    if (!training || p <= 0.0f)
        return result;

    Matrix mask(value.rows(), value.cols()); // the same as input, now initialized as zeros only
    std::mt19937 gen(42); // for now hardcoded seed
    std::bernoulli_distribution dist(p);
    mask.apply_inplace([&](float) { return dist(gen) ? 1.0f : 0.0f; });

    // dropout is applying 1/0 mask to input
    result.value = value.elementwise_multiply(mask) * (1.0f / p);
    // applying the same mask to backward signal
    if (requires_grad) {
        result.parents = { this };
        result.backward_fn = [this, mask, p](const Tensor &self) mutable {
            if (this->requires_grad) {
                this->grad = this->grad + self.grad.elementwise_multiply(mask) * (1.0f / p);
            }
        };
    };
    return result;
};