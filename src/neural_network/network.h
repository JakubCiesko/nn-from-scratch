//
// Created by jakub-ciesko on 11/10/25.
//

#ifndef NETWORK_H
#define NETWORK_H
#include <deque>
#include "tensor.h"
#include <vector>
#include <memory>


class Network {
    public:
    Network(const std::vector<int> &layer_sizes, bool use_dropout, float dropout_p=0.5, int seed=42);
    [[nodiscard]] std::vector<std::shared_ptr<Tensor>> &get_params()  {return params_;};
    // returns logits, training flag is used for dropout if use_dropout set to true
    Tensor forward(const Tensor& X, bool training) const ;

private:
    bool use_dropout_;
    float dropout_p_;
    std::vector<std::shared_ptr<Tensor>> params_;
    // this deque will keep all tensors alive until backward()
    mutable std::deque<Tensor> tape;
};

#endif //NETWORK_H