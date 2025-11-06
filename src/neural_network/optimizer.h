//
// Created by jakub-ciesko on 11/6/25.
//

#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "tensor.h"
#include <memory>
#include <vector>

class Optimizer
{
  public:
    Optimizer(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate);
    void zero_grad();
    void step();

  private:
    std::vector<std::shared_ptr<Tensor>> params;
    float lr;
};

#endif // OPTIMIZER_H