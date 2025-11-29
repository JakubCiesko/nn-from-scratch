//
// Created by jakub-ciesko on 9/23/25.
//
#ifndef MATRIX_H
#define MATRIX_H
#include <functional>
#include <vector>
#include <string>
#include <random>
/**
* Simple Matrix class used for linear algebra inside neural network implementation. Elements are of type float.
*/
class Matrix
{
  public:
    // init methods for matrix (what data will be in a matrix by default)
    enum class InitMethod
    {
        ZERO, // all zeros
        ONE, // all ones
        NORMAL, // drawn from normal distribution
        KAIMING // kaiming = normal(0, sqrt(2/# input)) page 4: https://arxiv.org/pdf/1502.01852 this is good for relu which I will be using solely
     };
    /** Constructor. @param rows Number of rows. @param cols Number of columns. @param method Initialization method. */
    Matrix(int rows, int cols, InitMethod method = InitMethod::ZERO, std::mt19937* gen = nullptr);
    /** helper method to visualize matrix content */
    void print() const;
    [[nodiscard]] int rows() const;
    [[nodiscard]] int cols() const;
    // basic arithmetic
    Matrix operator+(const Matrix &other) const;
    Matrix operator-(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;
    Matrix operator+(float scalar) const;
    Matrix operator-(float scalar) const;
    Matrix operator*(float scalar) const;
    void matmul(const Matrix &other, Matrix &result) const;
    // set get
    float &operator()(int row, int col);
    float operator()(int row, int col) const;
    void set(int row, int col, float value);
    [[nodiscard]] float get(int row, int col) const;
    // matrix manipulation
    /** transposes matrix inplace, does not create a new transposed matrix */
    void transpose_inplace();
    /** Returns transposed matrix. */
    [[nodiscard]] Matrix transpose() const;
    // matrix operation
    /** Applies function to all elements of matrix (in-place). */
    void apply_inplace(const std::function<float(float)>& function);
    [[nodiscard]] Matrix apply(const std::function<float(float)>& function) const;
    // all axis arguments: axis=0 over rows, axis=1 over columns.
    [[nodiscard]] Matrix sum_over(int axis) const;
    [[nodiscard]] Matrix mean_over(int axis) const;
    [[nodiscard]] Matrix std_over(int axis) const;
    [[nodiscard]] Matrix max_over(int axis) const;
    /** Haddamard product of two matrices */
    [[nodiscard]] Matrix elementwise_multiply(const Matrix &other) const;
    // broadcast operations are mainly used for adding bias vectors (Wx + b) in NN layers
    /** Broadcast addition over given axis */
    [[nodiscard]] Matrix broadcast_add(const Matrix &other, int axis) const;
    [[nodiscard]] Matrix broadcast_divide(const Matrix &other, int axis) const;
    [[nodiscard]] Matrix
    argmax(int axis) const; // will use for gettint predictions out of logits
    /** Simple "fused" method which performs matmul and broadcast addition in one pass to speed up essential Wx + b calculation in NN */
    [[nodiscard]] Matrix matmul_broadcast_add(const Matrix &B, const Matrix &C) const;
    void matmul_broadcast_add_prealloc(const Matrix &B, const Matrix &C, Matrix &result) const;

  private:
    /** Initializes matrix elements. */
    void initialize(InitMethod method, std::mt19937* gen);
    int rows_;
    int cols_;
    // data is float
    std::vector<float> data_;

    void check_dims_match_(const Matrix &other) const; // used for checking whether two matrices shapes match
    void check_dims_matmul_(const Matrix &other) const; // helper for checking matmul conditions
    [[nodiscard]] std::string shape_str_() const; // used in helpers dims checkers
    void check_element_indices_(int row, int col) const;
};

#endif
