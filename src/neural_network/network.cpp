//
// Created by jakub-ciesko on 11/10/25.
//
#include "network.h"
#include <iostream>
#include <stdexcept>
#include <random>


Network::Network(const std::vector<int> &layer_sizes, bool use_dropout, float dropout_p, int seed):
    use_dropout_(use_dropout), dropout_p_(dropout_p){
    std::mt19937 gen(seed);
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("Not enough layers. Need at least 2.");
    }
    for (size_t i = 0; i < layer_sizes.size() - 1; i++) {
        int in_dim = layer_sizes[i];
        int out_dim = layer_sizes[i + 1];

        Matrix W_val(in_dim, out_dim, Matrix::InitMethod::KAIMING, &gen);
        auto W = std::make_shared<Tensor>(W_val, true);
        Matrix b_val(1, out_dim, Matrix::InitMethod::ZERO, &gen);
        auto b = std::make_shared<Tensor>(b_val, true);

        params_.push_back(W);
        params_.push_back(b);
    }
}


Tensor Network::forward(const Tensor& X, bool training) const {
    // the tape is rebuilt on each forward pass
    tape.clear();
    tape.push_back(X);

    for (size_t i = 0; i < params_.size(); i += 2) {
        auto& W = *params_[i];
        auto& b = *params_[i + 1];

        // input to layer at the end of tape
        Tensor& input = tape.back();

        // linear layer output to tape
        tape.push_back(input.matmul_broadcast_add(W, b));

        // Check if hidden layer (not last)
        if (i + 2 < params_.size()) {
            Tensor& z = tape.back();


            tape.push_back(z.relu());

            // applying dropout mask
            if (use_dropout_) {
                Tensor& activated = tape.back();
                tape.push_back(activated.dropout(dropout_p_, training));
            }
        }
    }

    return tape.back();
}
