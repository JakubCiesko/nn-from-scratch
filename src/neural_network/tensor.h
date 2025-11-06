//
// Created by jakub-ciesko on 10/6/25.
//

// Tensor is generalization of matrix, scalars, vectors, matrices, 3D matrices, ... all are tensors
// tensors flow, are connected and contain their value (Matrix) and gradient. Some tensors do not need to have grad

#ifndef TENSOR_H
#define TENSOR_H
#include "matrix.h"

class Tensor {
    public:
        Tensor(const Matrix& value, bool requires_grad = false);
        Matrix value;
        Matrix grad;
        bool requires_grad;
        std::function<void()> backward_fn;
        void backward();
        void zero_grad();
        Tensor operator+(const Tensor&) const;
        Tensor operator-(const Tensor&) const;
        Tensor elementwise_multiply(const Tensor&) const;


    private:
        //nothing yet?
};


#endif //TENSOR_H