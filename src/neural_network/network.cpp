//
// Created by jakub-ciesko on 11/10/25.
//
#include "network.h"
#include <iostream>
#include <stdexcept>
#include <random>

/**
 * Network init code. Generates parameters of network (weights and biases) and makes them trainable.
 * Sets initialization methods of W and b according to: https://cs231n.github.io/neural-networks-2/
 * and https://en.wikipedia.org/wiki/Weight_initialization (He initialization). Sets dropout flag
 * and dropout_probability
 */
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
/**
 * Runs one forward pass using Tensors which are created on the fly and destroyed after.
 */
Tensor Network::forward(const Tensor& X, const bool training) const {
    // inference with torch.no_grad()
    if (!training) {
        // no need for storing, just reassigning current value
        Tensor current = X;
        current.requires_grad = false;
        for (size_t i = 0; i < params_.size(); i += 2) {
            auto& W = *params_[i];
            auto& b = *params_[i + 1];
            current = current.matmul_broadcast_add(W, b);
            // NO DROPOUT OUTSIDE OF TRAINING
            if (i + 2 < params_.size()) {
                current = current.relu();
            }
        }
        return current;
    }

    //training
    tape.clear(); // tape of tensors is rebuilt on each call of method
    tape.push_back(X);

    for (size_t i = 0; i < params_.size(); i += 2) {
        auto& W = *params_[i];
        auto& b = *params_[i + 1];
        Tensor& input = tape.back();
        tape.push_back(input.matmul_broadcast_add(W, b));
        if (i + 2 < params_.size()) {
            Tensor& z = tape.back();
            tape.push_back(z.relu());
            // dropout applied to every other layer
            size_t layer_index = i / 2;
            if (use_dropout_ && (layer_index % 2 == 0)) {
                Tensor& activated = tape.back();
                tape.push_back(activated.dropout(dropout_p_, true));
            }
        }
    }
    return tape.back();
}


/**
 * Runs one forward pass using preallocated resources.
 */
Tensor Network::forward_prealloc( Tensor& X, const bool training) const {
    // if no preallocated memory, we need initialization
    bool needs_init = state_cache_.empty();

    // different batch sizes might need reinitialization too
    if (!needs_init) {
        if (state_cache_[0].value.rows() != X.value.rows()) {
            needs_init = true;
            const_cast<Network*>(this)->state_cache_.clear();
        }
    }

    if (needs_init) {
        // memory reservation -- 3x # parameters (layer, relu, dropout)
        const_cast<Network*>(this)->state_cache_.reserve(params_.size() * 3);
    }

    size_t cache_idx = 0;
    Tensor* current_input = &X;

    for (size_t i = 0; i < params_.size(); i += 2) {
        auto& W = *params_[i];
        auto& b = *params_[i + 1];

        if (needs_init) {
            // Allocation of memory for tensors
            const_cast<Network*>(this)->state_cache_.push_back(
                current_input->matmul_broadcast_add(W, b)
            );
        } else {
            // here no need for allocation, it is already present in the cache
            Tensor& output_buffer = const_cast<Network*>(this)->state_cache_[cache_idx];
            current_input->matmul_broadcast_add_prealloc(W, b, output_buffer);
        }

        current_input = &state_cache_[cache_idx];
        cache_idx++;


        if (i + 2 < params_.size()) {
            // relu
            if (needs_init) {
                const_cast<Network*>(this)->state_cache_.push_back(current_input->relu());
            } else {
                Tensor& relu_buffer = const_cast<Network*>(this)->state_cache_[cache_idx];
                current_input->relu_prealloc(relu_buffer);
            }

            current_input = &state_cache_[cache_idx];
            cache_idx++;

            // dropout
            size_t layer_index = i / 2;
            bool apply_dropout = use_dropout_ && training && (layer_index % 2 == 0);

            if (apply_dropout) {
                if (needs_init) {
                     const_cast<Network*>(this)->state_cache_.push_back(
                         current_input->dropout(dropout_p_, training)
                     );
                } else {
                    Tensor& drop_buffer = const_cast<Network*>(this)->state_cache_[cache_idx];
                    drop_buffer = current_input->dropout(dropout_p_, training);
                }
                current_input = &state_cache_[cache_idx];
                cache_idx++;
            }
        }
    }
    return state_cache_.back();
};

/**
 * Loops through all Tensors which are used as preallocated resources and zeros out the gradients
 */
void Network::zero_grad_cache() {
    for (auto& t : state_cache_) {
        if (t.grad.rows() > 0) {
            t.grad.fill(0.0f);
        }
    }
}