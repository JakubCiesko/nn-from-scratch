#include <chrono>

#include "src/data/data_preparator.h"
#include "src/neural_network/matrix.h"
#include "src/neural_network/optimizer.h"
#include "src/neural_network/tensor.h"
#include "src/neural_network/network.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <memory>
#include <vector>

struct TrainingParams {
    int epochs;
    int batch_size;
    float adam_lr;
    float adam_beta1;
    float adam_beta2;
};
std::string current_time();
float compute_accuracy(const Matrix &logits, const Matrix &y_true);

void prepare_data(DataPreparator &data_preparator, bool standardize_data);
void train(TrainingParams training_params, Network &network,
    Optimizer &optimizer, DataPreparator &data_preparator);
void predict(Network &network, DataPreparator &data_preparator,
    bool is_test, const std::string &filename);

int main()
{
    auto start_time = std::chrono::steady_clock::now();
    int seed = 42;
    TrainingParams training_params = {
        2,
        128, // 128, 64 and 32 were good, but choppy loss
        0.0008f,
        0.9f,
        0.999f,
    };

    DataPreparator data_preparator("../data/", seed, training_params.batch_size);
    prepare_data(data_preparator, true);

    // model definition
    std::vector<int> architecture = {28 * 28, 64, 32, 10};
    Network network(architecture, seed);
    // optimizer definition
    AdamOptimizer optimizer(network.get_params(), training_params.adam_lr,
        training_params.adam_beta1, training_params.adam_beta2, 1e-8f);

    train(training_params, network, optimizer, data_preparator);
    predict(network, data_preparator, true, "../test_predictions.csv");

    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Total code run time: " << duration.count() << " seconds" << std::endl;
    return 0;
}

std::string current_time()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t now_c = system_clock::to_time_t(now);
    std::tm *parts = std::localtime(&now_c);

    std::ostringstream oss;
    oss << std::put_time(parts, "%H:%M:%S");
    return oss.str();
}

void prepare_data(DataPreparator &data_preparator, bool standardize_data) {
    data_preparator.load_data();
    if (standardize_data)
        data_preparator.standardize_data();
}


void train(TrainingParams training_params, Network &network, Optimizer &optimizer, DataPreparator &data_preparator) {

    Matrix loss_val(1, 1);
    auto network_params = network.get_params();
    // now just for loop instead of the below provided thing ?
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
    Tensor loss(Matrix(1,1), true);
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

            // forward pass
            // forward pass (fused matmul + bias for speed)
            Tensor y1_linear = X_batch.matmul_broadcast_add(*W1, *b1);
            Tensor y1 = y1_linear.relu();

            Tensor y2_linear = y1.matmul_broadcast_add(*W2, *b2);
            Tensor y2 = y2_linear.relu();

            Tensor logits = y2.matmul_broadcast_add(*W3, *b3);
            Tensor loss = logits.cross_entropy_loss(y_batch);
            loss_val = loss.value;

            // backprop
            loss.backward();
            // parameter updates
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
    Matrix X_data = is_test ? data_preparator.get_X_test() : data_preparator.get_X_train();
    Matrix y_data = is_test ? data_preparator.get_y_test() : data_preparator.get_y_train();
    Tensor X(X_data);
    Tensor logits = network.forward(X);
    float acc = compute_accuracy(logits.value, y_data);
    // this takes away from time limit, but it is nice feature
    std::cout << data_name << " Accuracy: " << acc << std::endl;
    data_preparator.save_predictions(logits.value, filename);
}


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
