//
// Created by jakub-ciesko on 10/6/25.
//

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

    private:
        //nothing yet?
};


#endif //TENSOR_H