//
// Created by jakub-ciesko on 11/6/25.
//
#include "optimizer.h"
#include <map>
#include <cmath>

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


AdamOptimizer::AdamOptimizer(const std::vector<std::shared_ptr<Tensor> > &params, float learning_rate, float beta1, float beta2, float epsilon):
    Optimizer(params, learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), timestep(0){
    initialize_momentum();
    initialize_velocity();
};

void AdamOptimizer::initialize_momentum() {
    for (auto &param : params)
        m.push_back(Matrix(param->value.rows(), param->value.cols(), Matrix::InitMethod::ZERO));
}

void AdamOptimizer::initialize_velocity() {
    for (auto &param : params)
        v.push_back(Matrix(param->value.rows(), param->value.cols(), Matrix::InitMethod::ZERO));
}



void AdamOptimizer::step()
{
    // source of formulas: https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
    timestep++;
    for (size_t i = 0; i < params.size(); ++i)
    {
        auto& param = params[i];
        if (param->requires_grad) {
            m[i] = m[i] * beta1 + (param -> grad )* (1.0f - beta1);
            v[i] = v[i] * beta2 + (param -> grad).elementwise_multiply(param -> grad) * (1.0f - beta2);
            Matrix m_correction = m[i] * (1.0f / (1.0f - powf(beta1, timestep)));
            Matrix v_correction = v[i] * (1.0f / (1.0f - powf(beta2, timestep)));
            param->value = param->value - m_correction.elementwise_multiply(
            v_correction.apply([this](const float el){ return 1.0f / (sqrtf(el) + epsilon); })) * lr;
        }
    }
}

AdamWOptimizer::AdamWOptimizer(const std::vector<std::shared_ptr<Tensor> > &params, float learning_rate, float beta1, float beta2, float weight_decay, float epsilon):
    AdamOptimizer(params, learning_rate, beta1, beta2, epsilon), weight_decay(weight_decay){};


void AdamWOptimizer::step() {
    // formulas from https://optimization.cbe.cornell.edu/index.php?title=AdamW
    // this is basically the same as Adam optim. only that there is weight decay applied to parameter values
    timestep++;
    for (size_t i = 0; i < params.size(); ++i)
    {
        auto& param = params[i];
        if (param->requires_grad) {
            m[i] = m[i] * beta1 + (param -> grad )* (1.0f - beta1);
            v[i] = v[i] * beta2 + (param -> grad).elementwise_multiply(param -> grad) * (1.0f - beta2);
            Matrix m_correction = m[i] * (1.0f / (1.0f - powf(beta1, timestep)));
            Matrix v_correction = v[i] * (1.0f / (1.0f - powf(beta2, timestep)));
            param->value = param->value - m_correction.elementwise_multiply(
            v_correction.apply([this](const float el){ return 1.0f / (sqrtf(el) + epsilon); })) * lr - param->value*weight_decay*lr;
        }
    }
}
