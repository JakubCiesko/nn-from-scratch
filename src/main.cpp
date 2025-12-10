#include <chrono>
#include <iostream>
#include <ostream>

#include "./utils/train.h"
#include "./data/data_preparator.h"
#include "./neural_network/network.h"
#include "./neural_network/optimizer.h"


int main() {

    const auto start_time  = std::chrono::high_resolution_clock::now();
    TrainingParams training_params = {
        10, //50
        256,
        1e-3f,
        0.9f,
        0.999f,
        1e-2f,
        1e-6f
    };

    DataPreparator data_preparator("./data/", 42, training_params.batch_size, "fashion_mnist");
    // prepare data (apply standardization)
    prepare_data(data_preparator, true);

    //28*28, 512, 256, 128, 10
    //28*28, 256, 128, 64, 10
    //28*28, 256, 128, 32, 10

    int input_dim = data_preparator.get_features_dim();
    int num_classes = 10;

    Network network({input_dim, 256, 128, 16, num_classes}, true, 0.4f, 42);
    AdamWOptimizer optimizer(network.get_params(), training_params.adam_lr,
                             training_params.adam_beta1, training_params.adam_beta2,
                             training_params.weight_decay, training_params.epsilon);

    train(training_params, network, optimizer, data_preparator);
    // sanity check of learning on training data
    predict(network, data_preparator, false, "./train_predictions.csv");
    // test data performance
    predict(network, data_preparator, true, "./test_predictions.csv");
    const auto end_time  = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    constexpr int second_limit = 600;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
    std::cout << "Time limit elapsed time: " << elapsed_seconds.count() << "/" << second_limit << " seconds" << std::endl;

}

