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
    virtual void step();

  protected:
    std::vector<std::shared_ptr<Tensor>> params;
    float lr;
};

class AdamOptimizer : public Optimizer {
public:
  AdamOptimizer(const std::vector<std::shared_ptr<Tensor>> &params, float learning_rate, float beta1, float beta2, float epsilon=1e-8);
  void step() override;
private:
  void initialize_momentum();
  void initialize_velocity();
  float beta1, beta2, epsilon;
  std::vector<Matrix> m; // momentum first moment
  std::vector<Matrix> v; // velocity second moment
  int timestep; // for bias correction in initial steps
};

#endif // OPTIMIZER_H