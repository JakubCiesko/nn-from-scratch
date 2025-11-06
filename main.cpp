#include "src/neural_network/matrix.h"
#include "src/neural_network/tensor.h"
#include <iomanip> // For std::setprecision
#include <iostream>

int main()
{
    std::cout << "--- Cross-Entropy Loss Autograd Test ---" << std::endl;
    std::cout << std::fixed << std::setprecision(5); // Format output

    // 1. Create Logits (Network Output)
    // We'll simulate a batch of 2, with 4 classes.
    Matrix logits_val(2, 4, Matrix::InitMethod::ZERO);
    // Deterministic values for logits
    logits_val.set(0, 0, 0.1f);
    logits_val.set(0, 1, 0.5f);
    logits_val.set(0, 2, 0.1f);
    logits_val.set(0, 3, 0.3f);
    logits_val.set(1, 0, 0.1f);
    logits_val.set(1, 1, -0.2f);
    logits_val.set(1, 2, -0.5f);
    logits_val.set(1, 3, 1.0f);

    Tensor logits(logits_val, true); // Logits MUST require grad

    // 2. Create True Labels
    // Your loss function expects raw labels (B, 1),
    // just like your DataPreparator provides.
    Matrix labels_val(2, 1, Matrix::InitMethod::ZERO);
    labels_val.set(0, 0, 1); // Row 0, correct class is 1
    labels_val.set(1, 0, 3); // Row 1, correct class is 3

    Tensor y_true(labels_val, false); // Labels do not require grad

    // --- 3. Forward Pass ---
    std::cout << "\n--- Forward Pass ---" << std::endl;
    // Calculate the loss
    Tensor loss = logits.cross_entropy_loss(y_true);

    std::cout << "Logits (Input):" << std::endl;
    logits.value.print();
    std::cout << "True Labels:" << std::endl;
    y_true.value.print();
    std::cout << "Calculated Loss:" << std::endl;
    loss.value.print();

    // --- 4. Backward Pass ---
    std::cout << "\n--- Backward Pass ---" << std::endl;
    // This will set loss.grad = 1.0 and start the chain
    loss.backward();

    std::cout << "Gradient of Loss (should be 1.0):" << std::endl;
    loss.grad.print();

    std::cout << "\nGradient of Logits (dLoss/dLogits):" << std::endl;
    logits.grad.print();

    // --- 5. Verify Results ---
    std::cout << "\n--- Expected Results ---" << std::endl;
    std::cout << "Expected Loss: 0.90426" << std::endl;
    std::cout << "Expected Logits Gradient:" << std::endl;
    std::cout << " 0.10603 -0.34177  0.10603  0.12971" << std::endl;
    std::cout << " 0.10543  0.07788  0.05786 -0.24117" << std::endl;

    return 0;
}