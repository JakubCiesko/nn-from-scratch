//
// Created by jakub-ciesko on 11/10/25.
//
#include "network.h"
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <random>

// helper to log results with time data
// std::string current_time()
// {
//     using namespace std::chrono;
//     auto now = system_clock::now();
//     std::time_t now_c = system_clock::to_time_t(now);
//     std::tm *parts = std::localtime(&now_c);
//
//     std::ostringstream oss;
//     oss << std::put_time(parts, "%H:%M:%S");
//     return oss.str();
// }

Network::Network(const std::vector<int> &layer_sizes, int seed) {
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


Tensor Network::forward(const Tensor &X) const {
    Tensor out = X;
    for (size_t i = 0; i < params_.size(); i += 2) {
        // apply linear layer of weight and bias
        auto &W = *params_[i];
        auto &b = *params_[i + 1];
        Tensor z = out.matmul_broadcast_add(W, b);
        // all but last layers use relu
        if (i + 2 < params_.size()) {
            out = z.relu();
        } else {
            out = z;
        }
    }
    return out;
}


//
// // I presuppose there is both bias and weights. If you do not want bias, set it to zero matrix non trainable
// Tensor Network::forward(Tensor &X) const {
//     Tensor out = X;
//     for (size_t i = 0; i < params_.size(); i += 2) {
//         auto &W = *params_[i];
//         auto &b = *params_[i + 1];
//         W.value.print();
//         b.value.print();
//         out = out.matmul_broadcast_add(W, b);
//         if (i + 2 < params_.size())
//             out = out.relu();
//     }
//     return out;
// }
//
// void Network::train(int epochs, DataPreparator& data_preparator, Optimizer& optimizer) {
//     Matrix loss_val(1, 1);
//     for (int e = 0; e < epochs; ++e)
//     {
//         data_preparator.reset_epoch();
//         int batch_i = 0;
//         while (data_preparator.has_next_batch())
//         {
//             if (++batch_i % 100 == 0)
//                 std::cout << "[" << current_time() << "] "
//                           << "[Epoch " + std::to_string(e + 1) +
//                                  "] Batch number: " + std::to_string(batch_i) +
//                                  " Loss: " + std::to_string(loss_val.get(0, 0))
//                           << std::endl;
//             optimizer.zero_grad(); // zero out all grads
//             auto [X_batch_mat, y_batch_mat] = data_preparator.get_batch();
//             Tensor X_batch(X_batch_mat, false);
//             Tensor y_batch(y_batch_mat, false);
//             Tensor logits = forward(X_batch);
//             logits.value.print();
//             Tensor loss = logits.cross_entropy_loss(y_batch);
//             loss_val = loss.value;
//             loss.backward();
//             optimizer.step();
//         }
//
//         if (e % 2 == 0 || e == epochs - 1)
//         {
//             std::cout << "[" << current_time() << "] " << "[Epoch" << std::setw(2)
//                       << e + 1 << "] Loss: ";
//             loss_val.print();
//         }
//     }
// }
//
// Matrix Network::test(DataPreparator &data_preparator) const {
//     Matrix X_test(data_preparator.get_X_test());
//     Matrix logits(X_test.rows(), 10);
//     for (size_t i = 0; i < params_.size(); i += 2) {
//         auto &W = params_[i]->value;
//         auto &b = params_[i + 1] -> value;
//         logits = logits.matmul_broadcast_add(W, b);
//         if (i + 2 < params_.size())
//             logits = logits.apply([](float val) { return std::max(0.0f, val); });
//     }
//     return logits;
// }