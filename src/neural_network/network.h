//
// Created by jakub-ciesko on 11/10/25.
//

#ifndef NETWORK_H
#define NETWORK_H
#include "tensor.h"
#include "matrix.h"
#include "optimizer.h"
#include <vector>

#include <memory>
#include "../data/data_preparator.h"

class Network {
    public:
    Network(const std::vector<int> &layer_sizes);
    [[nodiscard]] std::vector<std::shared_ptr<Tensor>> &get_params()  {return params_;};
    void train(int epochs, DataPreparator &data_preparator, Optimizer &optimizer);
    Tensor forward(Tensor &X) const;
    Matrix test(DataPreparator &data_preparator) const; // return logits
private:
    std::vector<std::shared_ptr<Tensor>> params_;
};

#endif //NETWORK_H