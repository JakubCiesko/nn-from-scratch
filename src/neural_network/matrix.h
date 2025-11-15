//
// Created by jakub-ciesko on 9/23/25.
//
#ifndef MATRIX_H
#define MATRIX_H
#include <functional>
#include <vector>
#include <string>

class Matrix
{
  public:
    enum class InitMethod
    {
        ZERO,
        ONE,
        NORMAL,
        KAIMING
    };
    Matrix(int rows, int cols, InitMethod method = InitMethod::ZERO);
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
    // set get
    float &operator()(int row, int col);
    float operator()(int row, int col) const;
    void set(int row, int col, float value);
    [[nodiscard]] float get(int row, int col) const;
    // matrix manipulation
    void transpose_inplace();
    [[nodiscard]] Matrix transpose() const;
    // matrix operation
    void apply_inplace(const std::function<float(float)>& function);
    [[nodiscard]] Matrix apply(const std::function<float(float)>& function) const;
    [[nodiscard]] Matrix sum_over(int axis) const;
    [[nodiscard]] Matrix mean_over(int axis) const;
    [[nodiscard]] Matrix std_over(int axis) const;
    [[nodiscard]] Matrix max_over(int axis) const;
    [[nodiscard]] Matrix elementwise_multiply(const Matrix &other) const;
    [[nodiscard]] Matrix broadcast_add(const Matrix &other, int axis) const;
    [[nodiscard]] Matrix broadcast_divide(const Matrix &other, int axis) const;
    [[nodiscard]] Matrix
    argmax(int axis) const; // will use for gettint predictions out of logits
    [[nodiscard]] Matrix matmul_broadcast_add(const Matrix &B, const Matrix &C) const;

  private:
    void initialize(InitMethod method);
    int rows_;
    int cols_;
    std::vector<float> data_;
    void check_dims_match_(const Matrix &other) const; // used for checking whether two matrices shapes match
    void check_dims_matmul_(const Matrix &other) const; // helper for checking matmul conditions
    [[nodiscard]] std::string shape_str_() const; // used in helpers dims checkers
    void check_element_indices_(int row, int col) const;
};

#endif
