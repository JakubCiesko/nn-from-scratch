//
// Created by jakub-ciesko on 12/5/25.
//
#include "train.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>

// helper function for displaying current time in logging
std::string current_time() {
    using namespace std::chrono;
    const auto now = system_clock::now();
    const std::time_t now_c = system_clock::to_time_t(now);
    const std::tm *parts = std::localtime(&now_c);

    std::ostringstream oss;
    oss << std::put_time(parts, "%H:%M:%S");
    return oss.str();
}

// accuracy computing -- used for sanity check of learning (train acc) and test set performance
float compute_accuracy(const Matrix &logits, const Matrix &y_true)
{
    int correct = 0;
    Matrix y_hat = logits.argmax(1);
    if (y_true.rows() != y_hat.rows())
        throw std::invalid_argument(
            "Invalid matrix row count for predictions and y_true");
    for (int i = 0; i < y_hat.rows(); i++)
    {
        int true_label = static_cast<int>(y_true(i, 0));
        int predicted_label = static_cast<int>(y_hat(i, 0));
        if (true_label == predicted_label)
            correct++;
    }
    return static_cast<float>(correct) / static_cast<float>(y_hat.rows());
};


// data preparation -- Loads csv files with training and test data, shuffling, standardization
void prepare_data(DataPreparator &data_preparator, const bool standardize_data) {
    data_preparator.load_data();
    if (standardize_data)
        data_preparator.standardize_data();
}

/**
 * the main training function. it goes through data in batches and trains the network using the provided optimizer
 */
void train(TrainingParams &training_params, Network &network, Optimizer &optimizer, DataPreparator &data_preparator) {

    Matrix loss_val(1, 1);
    std::cout << "[" << current_time() << "] "
              << "Starting training for " + std::to_string(training_params.epochs) + " epochs"
              << std::endl;

    // training loop over epochs over batches
    for (int e = 0; e < training_params.epochs; ++e)
    {

        int batch_i = 0;
        // restarts the datapreparator to start over from the beginning of data
        data_preparator.reset_epoch();

        while (data_preparator.has_next_batch())
        {
            if (++batch_i % 100 == 0)
                std::cout << "[" << current_time() << "] "
                          << "[Epoch " + std::to_string(e + 1) +
                                 "] Batch number: " + std::to_string(batch_i) +
                                 " Loss: " + std::to_string(loss_val.get(0, 0))
                          << std::endl;
            // zeroing out all the stored gradient
            optimizer.zero_grad();
            auto [X_batch_mat, y_batch_mat] = data_preparator.get_batch();

            // getting nontrainable tensors from data
            Tensor X_batch(X_batch_mat, false);
            Tensor y_batch(y_batch_mat, false);
            // logits are output of forward method
            Tensor y_hat = network.forward(X_batch, true);
            Tensor loss = y_hat.mse(y_batch);
            loss_val = loss.value;

            // backpropagation
            loss.backward();
            // param update
            optimizer.step();
        }

        if (e % 1 == 0 || e == training_params.epochs - 1)
        {
            std::cout << "[" << current_time() << "] " << "[Epoch" << std::setw(2)
                      << std::to_string(e + 1) << "] Loss: ";
            loss_val.print();
        }
    }
};

/**
 * the main training function. it goes through data in batches and trains the network using the provided optimizer.
 * It uses preallocated resources.
 */
void train_prealloc(TrainingParams &training_params, Network &network, Optimizer &optimizer, DataPreparator &data_preparator) {
    Matrix loss_val_mat(1, 1);
    Tensor loss_tensor(loss_val_mat, true);

    std::cout << "[" << current_time() << "] "
              << "Starting PREALLOC training for " + std::to_string(training_params.epochs) + " epochs"
              << std::endl;

    for (int e = 0; e < training_params.epochs; ++e)
    {
        int batch_i = 0;
        data_preparator.reset_epoch();

        while (data_preparator.has_next_batch())
        {
            auto [X_batch_mat, y_batch_mat] = data_preparator.get_batch();

            // in this case i skip the last batch of potentially malformed shape
            if (X_batch_mat.rows() != training_params.batch_size) {
                continue;
            }


            Tensor X_batch(X_batch_mat, false);
            Tensor y_batch(y_batch_mat, false);

            // zero out gradients which could be allocated across batches and make training explode
            optimizer.zero_grad();
            network.zero_grad_cache();

            // these two operations do not use preallocated memory
            // forward pass, returns final tensor value
            Tensor logits = network.forward_prealloc(X_batch, true);

            // loss calculation
            Tensor step_loss = logits.cross_entropy_loss(y_batch);

            if (++batch_i % 100 == 0) {
                 std::cout << "[" << current_time() << "] [Epoch " << (e + 1)
                           << "] Batch: " << batch_i
                           << " Loss: " << step_loss.value.get(0, 0) << std::endl;
            }

            // backpropagation of error singal
            step_loss.backward();
            // weight/params update
            optimizer.step();
        }
    }
};


/**
 * generates predictions, runs the whole test set in whole (not in batches).
 */

float MSE(const Matrix &y_hat, const Matrix &y_true) {
    Matrix diff = (y_hat - y_true);
    diff.apply_inplace([](const float x) {return x*x;});
    return diff.mean_over(0)(0,0);
}

void predict(Network &network, DataPreparator &data_preparator,
                          bool is_test, const std::string &filename) {

    const std::string data_name =  is_test? "Test Data" : "Train Data";
    std::cout << "[" << current_time() << "] Generating" << data_name << " predictions to: " << filename << std::endl;
    const Matrix X_data = is_test ? data_preparator.get_X_test() : data_preparator.get_X_train();
    const Matrix y_data = is_test ? data_preparator.get_y_test() : data_preparator.get_y_train();
    const Tensor X(X_data);
    const Tensor y_hat = network.forward(X, false);
    const float acc = MSE(y_hat.value, y_data);
    // this takes away from time limit, but it is nice feature
    std::cout << data_name << " Accuracy: " << acc << std::endl;
    data_preparator.save_predictions(y_hat.value, filename);
    Matrix my_guess(5, 1);
    my_guess(0,0) = 2.0;
    my_guess(1,0) = 3.0;
    my_guess(2,0) = 4.0;
    my_guess(3,0) = 5.0;
    my_guess(4,0) = 6.0;
    Tensor yy = network.forward(Tensor(my_guess, false), false);
    yy.value.print();
}
