#include <chrono>

#include "src/data/data_preparator.h"
#include "src/neural_network/matrix.h"
#include "src/neural_network/optimizer.h"
#include "src/neural_network/tensor.h"

#include <cmath>   // For sqrtf
#include <iomanip> // For std::setprecision
#include <iostream>
#include <memory> // For std::shared_ptr
#include <vector>

std::string current_time()
{
    using namespace std::chrono;
    auto now = system_clock::now();
    std::time_t now_c = system_clock::to_time_t(now);
    std::tm *parts = std::localtime(&now_c);

    std::ostringstream oss;
    oss << std::put_time(parts, "%H:%M:%S"); // Format as HH:MM:SS
    return oss.str();
}

float compute_accuracy(const Matrix &logits, const Matrix &y_true);

int main()
{
    auto start_time = std::chrono::steady_clock::now();
    int batch_size = 64;
    int seed = 42;
    DataPreparator data_preparator("./data/", seed, batch_size);
    data_preparator.load_data();
    data_preparator.standardize_data();

    Matrix X_train = data_preparator.get_X_train();
    Matrix y_train = data_preparator.get_y_train();

    // model definition
    int hidden_dim = 64;
    Matrix W1_val = (Matrix(28 * 28, hidden_dim, Matrix::InitMethod::KAIMING));
    Matrix b1_val(1, hidden_dim, Matrix::InitMethod::ZERO);
    auto W1 = std::make_shared<Tensor>(W1_val, true);
    auto b1 = std::make_shared<Tensor>(b1_val, true);

    Matrix W2_val = (Matrix(hidden_dim, hidden_dim * 2, Matrix::InitMethod::KAIMING));
    Matrix b2_val(1, hidden_dim * 2, Matrix::InitMethod::ZERO);
    auto W2 = std::make_shared<Tensor>(W2_val, true);
    auto b2 = std::make_shared<Tensor>(b2_val, true);

    Matrix W3_val = (Matrix(hidden_dim * 2, 10, Matrix::InitMethod::KAIMING));
    Matrix b3_val(1, 10, Matrix::InitMethod::ZERO);
    auto W3 = std::make_shared<Tensor>(W3_val, true);
    auto b3 = std::make_shared<Tensor>(b3_val, true);

    std::vector model_params{W1, b1, W2, b2, W3, b3};
    Optimizer optimizer(model_params, 0.01f);
    Matrix loss_val(1, 1);

    int epochs = 5;

    std::cout << "[" << current_time() << "] "
              << "Starting training for " + std::to_string(epochs) + " epochs"
              << std::endl;

    for (int e = 0; e < epochs; ++e)
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
            Tensor linear1 = X_batch * (*W1);
            Tensor linear1_bias = linear1.broadcast_add(*b1, 0);
            Tensor y1 = linear1_bias.relu();

            Tensor linear2 = y1 * (*W2);
            Tensor linear2_bias = linear2.broadcast_add(*b2, 0);
            Tensor y2 = linear2_bias.relu();

            Tensor linear3 = y2 * (*W3);
            Tensor logits = linear3.broadcast_add(*b3, 0); // this is logits

            Tensor loss = logits.cross_entropy_loss(y_batch);
            loss_val = loss.value;
            // backprop
            loss.backward();
            optimizer.step();
        }

        if (e % 1 == 0 || e == epochs - 1)
        {
            std::cout << "[" << current_time() << "] " << "[Epoch" << std::setw(2)
                      << e + 1 << "] Loss: ";
            loss_val.print();
        }
    }
    std::cout << "[" << current_time() << "] " << "Testing" << std::endl;
    Matrix X_test(data_preparator.get_X_test());
    // wild thing:
    Matrix z1 = (X_test * W1->value).broadcast_add(b1->value, 0);
    Matrix a1 = z1.apply([](float val) { return std::max(0.0f, val); });
    Matrix z2 = (a1 * W2->value).broadcast_add(b2->value, 0);
    Matrix a2 = z2.apply([](float val) { return std::max(0.0f, val); });
    Matrix logits = (a2 * W3->value).broadcast_add(b3->value, 0);
    float acc = compute_accuracy(logits, data_preparator.get_y_test());
    std::cout << "[" << current_time() << "] " << "TEST_ACC: " << acc << std::endl;
    auto end_time = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "Total code run time: " << duration.count() << " seconds" << std::endl;
    return 0;
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
}
