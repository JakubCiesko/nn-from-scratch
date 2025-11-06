#include "src/data/data_preparator.h"
#include "src/neural_network/matrix.h"
#include "src/neural_network/optimizer.h"
#include "src/neural_network/tensor.h"

#include <cmath>   // For sqrtf
#include <iomanip> // For std::setprecision
#include <iostream>
#include <memory> // For std::shared_ptr
#include <vector>

void xor_problem()
{
    std::cout << "--- XOR Training Test (Manual Tensors) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(5);

    // 1. Create XOR Data
    Matrix X_val(4, 2); // 4 samples, 2 features
    X_val.set(0, 0, 0.0f);
    X_val.set(0, 1, 0.0f);
    X_val.set(1, 0, 0.0f);
    X_val.set(1, 1, 1.0f);
    X_val.set(2, 0, 1.0f);
    X_val.set(2, 1, 0.0f);
    X_val.set(3, 0, 1.0f);
    X_val.set(3, 1, 1.0f);
    Tensor X(X_val, false);

    // Labels for 2-class cross-entropy must be (B, 1) with class indices
    Matrix y_val(4, 1); // 4 samples, 1 label
    y_val.set(0, 0, 0); // 0 ^ 0 = 0
    y_val.set(1, 0, 1); // 0 ^ 1 = 1
    y_val.set(2, 0, 1); // 1 ^ 0 = 1
    y_val.set(3, 0, 0); // 1 ^ 1 = 0
    Tensor y(y_val, false);

    // 2. Define Model Parameters Manually
    // We use std::shared_ptr so the optimizer can hold references
    // to them and they stay alive.

    // Layer 1: 2 inputs, 4 hidden units
    float stddev1 = sqrtf(2.0f / 2.0f); // Kaiming init
    Matrix W1_val = (Matrix(2, 4, Matrix::InitMethod::NORMAL) * stddev1);
    Matrix b1_val(1, 4, Matrix::InitMethod::ZERO);
    auto W1 = std::make_shared<Tensor>(W1_val, true);
    auto b1 = std::make_shared<Tensor>(b1_val, true);

    // Layer 2: 4 hidden units, 2 outputs (logits for class 0, class 1)
    float stddev2 = sqrtf(2.0f / 4.0f); // Kaiming init
    Matrix W2_val = (Matrix(4, 2, Matrix::InitMethod::NORMAL) * stddev2);
    Matrix b2_val(1, 2, Matrix::InitMethod::ZERO);
    auto W2 = std::make_shared<Tensor>(W2_val, true);
    auto b2 = std::make_shared<Tensor>(b2_val, true);

    // 3. Define Optimizer
    // Collect all parameters for the optimizer
    std::vector<std::shared_ptr<Tensor>> params = {W1, b1, W2, b2};
    float learning_rate = 0.1f;
    Optimizer optimizer(params, learning_rate);

    // 4. Training Loop
    int epochs = 1000;
    for (int e = 0; e < epochs; ++e)
    {
        // Zero gradients from previous step
        optimizer.zero_grad();

        // --- Manual Forward Pass ---
        // We must keep all intermediate Tensors (z1, a1, logits) in
        // scope so their backward_fn() can access valid data.

        // Layer 1
        Tensor matmul1 = X * (*W1);
        Tensor z1 = matmul1.broadcast_add(*b1, 0);
        Tensor a1 = z1.relu();

        // Layer 2
        Tensor matmul2 = a1 * (*W2);
        Tensor logits = matmul2.broadcast_add(*b2, 0);

        // --- Loss ---
        Tensor loss = logits.cross_entropy_loss(y);

        // --- Backward Pass ---
        // This computes gradients for loss, logits, a1, z1, W1, b1, W2, b2
        loss.backward();

        // --- Optimizer Step ---
        // This updates W1->value, b1->value, etc.
        optimizer.step();

        if (e % 100 == 0 || e == epochs - 1)
        {
            std::cout << "Epoch " << std::setw(4) << e << " Loss: ";
            loss.value.print();
        }
    }

    // 5. Show final predictions
    std::cout << "\n--- Final Predictions ---" << std::endl;
    std::cout << "Inputs (X):" << std::endl;
    X.value.print();

    std::cout << "True Labels (y):" << std::endl;
    y.value.print();

    // Run forward pass one last time
    Tensor z1 = (X * (*W1)).broadcast_add(*b1, 0);
    Tensor a1 = z1.relu();
    Tensor final_logits = (a1 * (*W2)).broadcast_add(*b2, 0);

    std::cout << "Final Logits (raw network output):" << std::endl;
    final_logits.value.print();

    std::cout << "\n--- Interpretation ---" << std::endl;
    std::cout << "(Logits are [class 0, class 1])" << std::endl;
    std::cout << "Input [0, 0] -> Expect Class 0 (e.g., [High, Low])" << std::endl;
    std::cout << "Input [0, 1] -> Expect Class 1 (e.g., [Low, High])" << std::endl;
    std::cout << "Input [1, 0] -> Expect Class 1 (e.g., [Low, High])" << std::endl;
    std::cout << "Input [1, 1] -> Expect Class 0 (e.g., [High, Low])" << std::endl;
}

float compute_accuracy(const Matrix &logits, const Matrix &y_true);

int main()
{
    int batch_size = 64;
    int seed = 42;
    DataPreparator data_preparator("../data/", seed, batch_size);
    data_preparator.load_data();
    data_preparator.standardize_data();

    Matrix X_train = data_preparator.get_X_train();
    Matrix y_train = data_preparator.get_y_train();

    // model definition
    int hidden_dim = 16;
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
    Optimizer optimizer(model_params, 0.001f);
    Matrix loss_val(1, 1);

    int epochs = 5;

    std::cout << "Starting training for " + std::to_string(epochs) + " epochs"
              << std::endl;

    for (int e = 0; e < epochs; ++e)
    {

        int batch_i = 0;
        data_preparator.reset_epoch();

        while (data_preparator.has_next_batch())
        {
            if (++batch_i % 100 == 0)
                std::cout << "[Epoch " + std::to_string(e + 1) +
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
            std::cout << "[Epoch" << std::setw(2) << e + 1 << "] Loss: ";
            loss_val.print();
        }
    }
    std::cout << "Testing" << std::endl;
    Matrix X_test(data_preparator.get_X_test());
    // wild thing:
    Matrix z1 = (X_test * W1->value).broadcast_add(b1->value, 0);
    Matrix a1 = z1.apply([](float val) { return std::max(0.0f, val); });
    Matrix z2 = (a1 * W2->value).broadcast_add(b2->value, 0);
    Matrix a2 = z2.apply([](float val) { return std::max(0.0f, val); });
    Matrix logits = (a2 * W3->value).broadcast_add(b3->value, 0);
    float acc = compute_accuracy(logits, data_preparator.get_y_test());
    std::cout << "TEST_ACC: " << acc << std::endl;
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
