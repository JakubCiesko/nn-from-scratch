//
// Created by jakub-ciesko on 9/23/25.
//
#ifndef MATRIX_H
#define MATRIX_H
#include <functional>
#include <vector>

class Matrix {
public:
    enum class InitMethod { ZERO, ONE, NORMAL };
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
    float& operator()(int row, int col);
    float operator()(int row, int col) const;
    void set(int row, int col, float value);
    [[nodiscard]] float get(int row, int col) const;
    // matrix manipulation
    void transpose_inplace();
    [[nodiscard]] Matrix transpose() const;
    // matrix operation
    Matrix apply(std::function<float(float)> function) const;
    void apply_inplace(std::function<float(float)> function);
    [[nodiscard]] Matrix sum_over(int axis) const;
    [[nodiscard]] Matrix mean_over(int axis) const;
    [[nodiscard]] Matrix std_over(int axis) const;

private:
    void initialize(InitMethod method);
    int rows_;
    int cols_;
    std::vector<float> data_;

};
#endif
