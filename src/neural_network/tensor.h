//
// Created by jakub-ciesko on 10/6/25.
//

// Tensor is generalization of matrix, scalars, vectors, matrices, 3D matrices,
// ... all are tensors tensors flow, are connected and contain their value
// (Matrix) and gradient. Some tensors do not need to have grad

#ifndef TENSOR_H
#define TENSOR_H

#include "matrix.h"
#include <vector>
class Tensor
{
  public:
    explicit Tensor(const Matrix &value, bool requires_grad = false);
    Matrix value;
    Matrix grad;
    bool requires_grad;
    std::function<void(const Tensor& self)> backward_fn;
    void backward();
    void zero_grad();
    [[nodiscard]] Tensor operator+(Tensor &);
    [[nodiscard]] Tensor operator-(Tensor &);
    [[nodiscard]] Tensor elementwise_multiply(Tensor &);
    [[nodiscard]] Tensor operator*(Tensor &);
    [[nodiscard]] Tensor relu();
    [[nodiscard]] Tensor cross_entropy_loss(const Tensor &y_true);
    [[nodiscard]] Tensor broadcast_add(Tensor &, int);
    [[nodiscard]] Tensor matmul_broadcast_add( Tensor &B, Tensor &C);
    void matmul(Tensor &B, Tensor &result);
    void matmul_broadcast_add_prealloc(Tensor &B, Tensor &C, Tensor &result);
    void relu_prealloc(Tensor &result);
    void relu_inplace();
    [[nodiscard]] Tensor dropout(float p, bool training);
    std::vector<Tensor*> parents;
};

#endif // TENSOR_H