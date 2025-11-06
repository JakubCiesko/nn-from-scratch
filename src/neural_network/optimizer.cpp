//
// Created by jakub-ciesko on 11/6/25.
//
#include "optimizer.h"

Optimizer::Optimizer(const std::vector<std::shared_ptr<Tensor>> &params,
                     float learning_rate)
    : params(params), lr(learning_rate){};

void Optimizer::step()
{
    for (auto &param : params)
    {
        if (param->requires_grad)
            param->value = param->value - (param->grad * lr);
    }
}

void Optimizer::zero_grad()
{
    for (auto &param : params)
    {
        if (param->requires_grad)
            param->zero_grad();
    }
}