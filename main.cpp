#include "src/neural_network/matrix.h"
#include "src/neural_network/tensor.h"
#include <iostream>

int main()
{
    std::cout << "--- Tensor Class Autograd Test ---" << std::endl;

    // 1. Create Tensors for a single layer: y = relu(x*w + b)
    // We'll simulate a batch of 2 samples (x), with 3 features each.
    // The layer will have 2 neurons.

    // x (input): 2x3 matrix (batch=2, features=3)
    Matrix x_val(2, 3, Matrix::InitMethod::NORMAL);
    Tensor x(x_val, true); // requires_grad = true

    // w (weights): 3x2 matrix (features=3, neurons=2)
    Matrix w_val(3, 2, Matrix::InitMethod::NORMAL);
    Tensor w(w_val, true); // requires_grad = true

    // b (bias): 1x2 matrix (1, neurons=2)
    Matrix b_val(1, 2, Matrix::InitMethod::ONE); // Start bias at 1.0
    Tensor b(b_val, true);                       // requires_grad = true

    // 2. --- Forward Pass ---
    std::cout << "\n--- Forward Pass ---" << std::endl;

    // z = x * w (MatMul)
    Tensor z = x * w; // z should be (2, 2)

    // a = z + b (Broadcast Add)
    Tensor a = z.broadcast_add(b, 0); // a should be (2, 2)

    // y = relu(a) (Activation)
    Tensor y = a.relu(); // y should be (2, 2)

    std::cout << "Input X (2x3):" << std::endl;
    x.value.print();
    std::cout << "Weights W (3x2):" << std::endl;
    w.value.print();
    std::cout << "Bias B (1x2):" << std::endl;
    b.value.print();
    std::cout << "Output Y (2x2):" << std::endl;
    y.value.print();

    // 3. --- Backward Pass ---
    std::cout << "\n--- Backward Pass ---" << std::endl;

    // To start backpropagation, we need an initial gradient.
    // In a real network, this comes from the loss function.
    // Here, we'll set the "incoming" gradient for 'y' to all 1s.
    y.grad = Matrix(2, 2, Matrix::InitMethod::ONE);

    // Call backward() on the *last* tensor in the chain.
    // This will trigger the chain reaction.
    y.backward();

    std::cout << "Gradient of Y (should be all 1s):" << std::endl;
    y.grad.print();

    std::cout << "\n--- Calculated Gradients ---" << std::endl;

    std::cout << "Gradient of Weights (W):" << std::endl;
    w.grad.print();

    std::cout << "Gradient of Input (X):" << std::endl;
    x.grad.print();

    std::cout << "Gradient of Bias (B):" << std::endl;
    b.grad.print();

    std::cout << "\n--- Test Complete ---" << std::endl;

    return 0;
}