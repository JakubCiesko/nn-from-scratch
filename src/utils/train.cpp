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
void prepare_data(DataPreparator &data_preparator, bool standardize_data) {
    data_preparator.load_data();
    if (standardize_data)
        data_preparator.standardize_data();
}

/*
 *
 * Architecture is hardcoded,
 */
void train(TrainingParams &training_params, Network &network, Optimizer &optimizer, DataPreparator &data_preparator) {

    Matrix loss_val(1, 1);
    auto network_params = network.get_params();
    // now just for loop instead of the below provided thing ?
    auto W1 = network.get_params()[0];
    auto b1 = network.get_params()[1];
    auto W2 = network.get_params()[2];
    auto b2 = network.get_params()[3];
    auto W3 = network.get_params()[4];
    auto b3 = network.get_params()[5];

    std::cout << "[" << current_time() << "] "
              << "Starting training for " + std::to_string(training_params.epochs) + " epochs"
              << std::endl;

    for (int e = 0; e < training_params.epochs; ++e)
    {

        int batch_i = 0;
        data_preparator.reset_epoch();

        while (data_preparator.has_next_batch())
        {
            if (++batch_i % 100 == 0)
                std::cout << "[" << current_time() << "] "
                          << "[Epoch " + std::to_string(e + 1) +
                                 "] Batch number: " + std::to_string(batch_i) +
                                 " Loss: " + std::to_string(loss_val.get(0, 0))
                          << std::endl;
            optimizer.zero_grad();
            auto [X_batch_mat, y_batch_mat] = data_preparator.get_batch();
            Tensor X_batch(X_batch_mat, false);
            Tensor y_batch(y_batch_mat, false);
            Tensor logits = network.forward(X_batch, true);
            Tensor loss = logits.cross_entropy_loss(y_batch);
            loss_val = loss.value;
            loss.backward();
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


/*
 * this training function uses prealloc methods in Tensor and Matrix classes.
 * It does not work well however, and the training stalls.
 */
void train_prealloc(TrainingParams training_params, Network &network, Optimizer &optimizer, DataPreparator &data_preparator) {

    Matrix loss_val(1, 1);
    auto network_params = network.get_params();

    auto W1 = network.get_params()[0];
    auto b1 = network.get_params()[1];
    auto W2 = network.get_params()[2];
    auto b2 = network.get_params()[3];
    auto W3 = network.get_params()[4];
    auto b3 = network.get_params()[5];

    Tensor y1_linear(Matrix(training_params.batch_size, W1->value.cols()), true);
    Tensor y1(Matrix(training_params.batch_size, W1->value.cols()), true);
    Tensor y2_linear(Matrix(training_params.batch_size,W2->value.cols()), true);
    Tensor y2(Matrix(training_params.batch_size,W2->value.cols()), true);
    Tensor logits(Matrix(training_params.batch_size, W3 -> value.cols()), true);

    std::cout << "[" << current_time() << "] "
              << "Starting training for " + std::to_string(training_params.epochs) + " epochs"
              << std::endl;

    for (int e = 0; e < training_params.epochs; ++e)
    {

        int batch_i = 0;
        data_preparator.reset_epoch();

        while (data_preparator.has_next_batch())
        {
            if (++batch_i % 100 == 0)
                std::cout << "[" << current_time() << "] "
                          << "[Epoch " + std::to_string(e + 1) +
                                 "] Batch number: " + std::to_string(batch_i) +
                                 " Loss: " + std::to_string(loss_val.get(0, 0))
                          << std::endl;
            optimizer.zero_grad();
            auto [X_batch_mat, y_batch_mat] = data_preparator.get_batch();
            if (X_batch_mat.rows() != training_params.batch_size)
                continue;

            Tensor X_batch(X_batch_mat, false);
            Tensor y_batch(y_batch_mat, false);

            //reset
            y1_linear.grad.apply_inplace([](float x) {return 0.0f;});
            y1_linear.parents.clear();
            y1_linear.backward_fn = nullptr;
            y1.grad.apply_inplace([](float x) {return 0.0f;});
            y1.parents.clear();
            y1.backward_fn = nullptr;
            y2_linear.grad.apply_inplace([](float x) {return 0.0f;});
            y2_linear.parents.clear();
            y2_linear.backward_fn = nullptr;
            y2.grad.apply_inplace([](float x) {return 0.0f;});
            y2.parents.clear();
            y2.backward_fn = nullptr;
            logits.grad.apply_inplace([](float x) {return 0.0f;});
            logits.parents.clear();
            logits.backward_fn = nullptr;


            X_batch.matmul_broadcast_add_prealloc(*W1, *b1, y1_linear);
            y1_linear.relu_prealloc(y1);

            y1.matmul_broadcast_add_prealloc(*W2, *b2, y2_linear);
            y2_linear.relu_prealloc(y2);

            y2.matmul_broadcast_add_prealloc(*W3, *b3, logits);

            Tensor loss = logits.cross_entropy_loss(y_batch);


            loss_val = loss.value;
            loss.backward();
            optimizer.step();

        }

        if (e % 1 == 0 || e == training_params.epochs - 1)
        {
            std::cout << "[" << current_time() << "] " << "[Epoch" << std::setw(2)
                      << e + 1 << "] Loss: ";
            loss_val.print();
        }
    }
};


void predict(Network &network, DataPreparator &data_preparator,
                          bool is_test, const std::string &filename) {

    const std::string data_name =  is_test? "Test Data" : "Train Data";
    std::cout << "[" << current_time() << "] Generating" << data_name << " predictions to: " << filename << std::endl;
    const Matrix X_data = is_test ? data_preparator.get_X_test() : data_preparator.get_X_train();
    const Matrix y_data = is_test ? data_preparator.get_y_test() : data_preparator.get_y_train();
    const Tensor X(X_data);
    const Tensor logits = network.forward(X, false);
    const float acc = compute_accuracy(logits.value, y_data);
    // this takes away from time limit, but it is nice feature
    std::cout << data_name << " Accuracy: " << acc << std::endl;
    data_preparator.save_predictions(logits.value, filename);
}
