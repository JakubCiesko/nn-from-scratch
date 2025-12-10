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

/**
 * Constructs a tensor with given value and gradient requirement.
 * @param value Initial value of the tensor.
 * @param requires_grad If true, gradients will be tracked for this tensor.
 */
Tensor::Tensor(const Matrix &value, bool requires_grad)
    : value(value), grad(value.rows(), value.cols(), Matrix::InitMethod::ZERO),
      requires_grad(requires_grad), backward_fn(nullptr){};
/**
 * Resets the gradient of this tensor to zero.
 */
void Tensor::zero_grad()
{
    grad.fill(0.0f);
}

/**
 * Performs backpropagation to compute gradients for all tensors in the computational graph.
 * Seeds gradient with 1 if the tensor is a scalar.
 * Uses a depth-first search to traverse the computational graph in topological order, then
 * applies the backward function in reverse order.
 */
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

/**
 * Operator + on Tensors, does matrix + matrix but also adds requires_grad to result if any of parents require grad.
 * Adds parents of that tensor.
 */
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
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + self.grad; // propagation of gradients
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

            }
            if (other.requires_grad)
            {
                other.grad = other.grad - self.grad; // propagation of gradients
            }
        };
    }
    return result;
}

/**
 * Performs elementwise multiplication between this tensor and another tensor.
 * Returns a new tensor representing the elementwise product.
 *
 * Backpropagation logic:
 * - The backward function computes gradients for each parent:
 *   - d(this)/d(result) = other.value * grad(result)
 *   - d(other)/d(result) = this->value * grad(result)
 */
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

            }

            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.elementwise_multiply(self.grad);
            }
        };
    }
    return result;
}

/**
 * Matmul with backprop
 */
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
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.transpose() * self.grad;
            }
        };
    }
    return result;
}

/**
 * broadcasted addition between this tensor and another tensor along the given axis. Returns a new tensor representing the result.
 *
 * Backpropagation:
 * - If this tensor requires gradients, the gradient is propagated directly (d(self)/d(this) = 1).
 * - If the other tensor requires gradients, the gradient is summed over the broadcasted axis
 *   to match its original shape (chain rule for broadcasting).
 */
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
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + self.grad.sum_over(axis);
            }
        };
    }
    return result;
}

/**
 * Returns a new tensor representing the result of applying relu to tensor.
 * Backpropagation is application of mask of 0s and 1s based on value of tensor,
 * this is then applied to backflowing gradient.
 */
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
            };
        };
    }
    return result;
}

/**
 * function to stabilize logit values -- shifts them row-wise by row maximum
 * to prevent overflow when calculating exponential
 */
Matrix stabilize_logits(const Matrix &logits)
{
    Matrix max_logits = logits.max_over(1);
    return logits.broadcast_add(max_logits * -1.0f, 1);
}


/**
 * Compute log-softmax row-wise.
 * Returns: log_softmax, exp(safe_logits), sum_exps
 */
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


/**
 * Cross-entropy loss for classification
 * Backprop: dL/dlogits = (softmax - y_true_one_hot) / batch_size
 */
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

    // cache softmax probabilities and true labels for use in backward pass
    auto softmax_probs_cache = std::move(softmax_probs);
    auto y_true_labels_cache = y_true.value;

    auto &logits_tensor = *this;
    // computational graph this tensor is parent of loss
    result.parents.push_back(this);
    result.backward_fn = [&logits_tensor, y_true_labels_cache, softmax_probs_cache,
                          num_classes](const Tensor &self) mutable
    {

        if (logits_tensor.requires_grad)
        {
            const int batch_size = logits_tensor.value.rows();

            // label to one hot
            Matrix y_true_one_hot(batch_size, num_classes, Matrix::InitMethod::ZERO);
            for (int i = 0; i < batch_size; ++i)
            {
                int correct_class = static_cast<int>(y_true_labels_cache.get(i, 0));
                y_true_one_hot.set(i, correct_class, 1.0f);
            }

            // gradient: softmax - one_hot
            Matrix grad_logits = softmax_probs_cache - y_true_one_hot;
            // average over batch
            Matrix mean_grad_logits =
                grad_logits * (1.0f / static_cast<float>(batch_size));
            // accumulate gradient into tensor
            logits_tensor.grad = logits_tensor.grad + mean_grad_logits;
        }
    };

    return result;
}


Tensor Tensor::mse(const Tensor &y_true)
{
    const Matrix &y_hat_val = this->value;
    const Matrix &y_true_val = y_true.value;

    const Matrix diff = y_hat_val - y_true_val;
    const Matrix sq_diff = diff.apply([](const float x){return x*x;});

    const float total_loss = sq_diff.mean_over(1)(0,0);
    Matrix result_value(1, 1, Matrix::InitMethod::ZERO);
    result_value.set(0, 0, total_loss);
    Tensor result(result_value, true);


    result.parents.push_back(this);
    auto &y_hat_tensor = *this;

    result.backward_fn = [&y_hat_tensor, diff](const Tensor &self) mutable
    {

        if (y_hat_tensor.requires_grad)
        {
            const int total_elems = diff.rows() * diff.cols();
            const float scale = 2.0f / static_cast<float>(total_elems);
            const Matrix mse_val = diff * scale;
            y_hat_tensor.grad = y_hat_tensor.grad + mse_val;
        }
    };

    return result;
}

/**
 * Matmul of this tensor with B and adds C (broadcasted). Fuses multiple operations into one using fused matrix method.
 * Faster than matmul + add.
 * Returns new tensor.
 */
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
            }
            if (B.requires_grad)
            {
                B.grad = B.grad + this->value.transpose() * self.grad;
            }
            if (C.requires_grad) {
                C.grad = C.grad + self.grad.sum_over(0);
            }
        };
    }
    return result;
}

/**
 * Matmul of tensors. Equivalent of operator* but uses preallocated tensor.
 */
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
            }
            if (other.requires_grad)
            {
                other.grad = other.grad + this->value.transpose() * self.grad;
            }
        };
    }

};

/**
 * Matmul of this tensor with B and adds C (broadcasted). Fuses multiple operations into one using fused matrix method.
 * Faster than matmul + add. Uses preallocated resources.
 * Returns new tensor.
 */
void Tensor::matmul_broadcast_add_prealloc(Tensor &B, Tensor &C, Tensor &result) {
    value.matmul_broadcast_add_prealloc(B.value, C.value, result.value);
    result.requires_grad = requires_grad || B.requires_grad || C.requires_grad ;
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
};

/**
 * Applies ReLU activation in-place.
 */
void Tensor::relu_inplace() {
    value.apply_inplace([](float x) { return std::max(0.0f, x); });
}


/**
 * Applies ReLU activation and stores result in preallocated tensor.
 */
void Tensor::relu_prealloc(Tensor &result) {
    std::function relu_fn = [](float val) { return std::max(0.0f, val); };
    result.value = value.apply(relu_fn);
    result.requires_grad = requires_grad;
    if (result.requires_grad)
    {
        result.parents = { this };
        result.backward_fn = [this](const Tensor &self) mutable
        {
            if (this->requires_grad)
            {
                const Matrix mask = this->value.apply([](float val)
                                                { return (val < 0.0f) ? 0.0f : 1.0f; });
                this->grad = this->grad + self.grad.elementwise_multiply(mask);
                //this->backward();
            };
        };
    }
}


/**
 * Applies dropout to tensor. Returns new tensor with dropped elements scaled.
 */
Tensor Tensor::dropout(const float p, const bool training=true) {
    // dropout is not to be used in test pass, also drop invalid probability values
    if (!training || p <= 0.0f)
        return *this;

    Matrix mask(value.rows(), value.cols()); // the same as input, now initialized as zeros only
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0f-p);
    mask.apply_inplace([&](float) { return dist(gen) ? 1.0f : 0.0f; });

    // dropout is applying 1/0 mask to input
    Tensor result = *this;
    const float scale_factor = 1.0f / (1.0f-p);
    result.value = value.elementwise_multiply(mask) * scale_factor;
    // applying the same mask to backward signal
    if (requires_grad) {
        result.parents = { this };
        result.backward_fn = [this, mask, scale_factor](const Tensor &self) mutable {
            if (this->requires_grad) {
                this->grad = this->grad + self.grad.elementwise_multiply(mask) * scale_factor;
            }
        };
    };
    return result;
};