//
// Created by jakub-ciesko on 9/23/25.
//
#include "matrix.h"
#include <iostream>
#include <random>
#include <stdexcept>

// init code
Matrix::Matrix(int rows, int cols, InitMethod method)
    : rows_(rows), cols_(cols), data_(rows * cols)
{
    initialize(method);
}

void Matrix::initialize(InitMethod method)
{
    if (method == InitMethod::ZERO)
    {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    else if (method == InitMethod::ONE)
    {
        std::fill(data_.begin(), data_.end(), 1.0f);
    }
    else if (method == InitMethod::NORMAL)
    {
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto &x : data_)
            x = dist(gen);
    }
    else if (method == InitMethod::KAIMING)
    {
        // TODO: FIX THIS!
        float stddev = sqrtf(2.0f / static_cast<float>(rows_));
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto &x : data_)
            x = dist(gen) * stddev;
    }
}

void Matrix::set(int row, int col, float value)
{
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_)
        throw std::out_of_range("Index out of range");
    data_[row * cols_ + col] = value;
}

float Matrix::get(int row, int col) const
{
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_)
        throw std::out_of_range("Index out of range");
    return data_[row * cols_ + col];
}

int Matrix::cols() const
{
    return cols_;
}

int Matrix::rows() const
{
    return rows_;
}

float &Matrix::operator()(int row, int col)
{
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_)
        throw std::out_of_range("Index out of range");
    return data_[row * cols_ + col];
}
float Matrix::operator()(int row, int col) const
{
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_)
        throw std::out_of_range("Index out of range");
    return data_[row * cols_ + col];
}

void Matrix::print() const
{
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            std::cout << data_[cols_ * i + j] << " ";
        }
        std::cout << std::endl;
    }
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("Matrix sizes must match for addition");
    }
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator+(float scalar) const
{
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = data_[i] + scalar;
    }
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (other.rows_ != rows_ || other.cols_ != cols_)
    {
        throw std::invalid_argument("Matrix sizes must match");
    }
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Matrix Matrix::operator-(float scalar) const
{
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = data_[i] - scalar;
    }
    return result;
}

// This will be enhanced
Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols_ != other.rows_)
        throw std::invalid_argument("Matrix sizes must match for multiplication");
    Matrix result(rows_, other.cols_, InitMethod::ZERO);
    // Cij = dot(Ai*, B*,j) = sum(AiK*bKj)
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < other.cols_; ++j)
        {
            for (int k = 0; k < cols_; ++k)
            {
                result.data_[i * other.cols_ + j] +=
                    data_[i * cols_ + k] * other.data_[k * other.cols_ + j];
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(float scalar) const
{
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

void Matrix::transpose_inplace()
{
    std::vector<float> transposed(data_.size());
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            transposed[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    data_ = std::move(transposed);
    int temp = rows_;
    rows_ = cols_;
    cols_ = temp;
}

Matrix Matrix::transpose() const
{
    Matrix result(cols_, rows_);
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < cols_; ++j)
        {
            result.data_[j * rows_ + i] = data_[i * cols_ + j];
        }
    }
    return result;
}

Matrix Matrix::apply(std::function<float(float)> function) const
{
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = function(data_[i]);
    }
    return result;
}

void Matrix::apply_inplace(std::function<float(float)> function)
{
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        data_[i] = function(data_[i]);
    }
}

Matrix Matrix::sum_over(int axis) const
{
    if (axis == 0)
    {
        // sum over rows, matrix is 1 * columns
        Matrix result(1, cols_, InitMethod::ZERO);
        for (int j = 0; j < cols_; ++j)
        {
            float sum = 0.0f;
            for (int i = 0; i < rows_; ++i)
            {
                sum += data_[i * cols_ + j];
            }
            result.data_[j] = sum;
        }
        return result;
    }
    if (axis == 1)
    {
        // Sum over columns: result is rows_ × 1
        Matrix result(rows_, 1, InitMethod::ZERO);
        for (int i = 0; i < rows_; ++i)
        {
            float sum = 0.0f;
            for (int j = 0; j < cols_; ++j)
            {
                sum += data_[i * cols_ + j];
            }
            result.data_[i] = sum;
        }
        return result;
    }
    throw std::invalid_argument("Invalid axis");
}

Matrix Matrix::mean_over(int axis) const
{
    if (axis == 0)
    {
        return sum_over(axis) * (1 / static_cast<float>(rows_));
    };
    if (axis == 1)
    {
        return sum_over(axis) * (1 / static_cast<float>(cols_));
    };
    throw std::invalid_argument("Invalid axis");
}

Matrix Matrix::std_over(int axis) const
{
    Matrix mean = mean_over(axis);
    if (axis == 0)
    {
        // sum over rows, matrix is 1 * columns
        Matrix std(1, cols_, InitMethod::ZERO);
        for (int j = 0; j < cols_; ++j)
        {
            float sum = 0.0f;
            for (int i = 0; i < rows_; ++i)
            {
                sum += powf((data_[i * cols_ + j] - mean(0, j)), 2.0f);
            }
            std.data_[j] = sqrtf(sum / static_cast<float>(rows_));
        }
        return std;
    }
    if (axis == 1)
    {
        // Sum over columns: result is rows_ × 1
        Matrix result(rows_, 1, InitMethod::ZERO);
        for (int i = 0; i < rows_; ++i)
        {
            float sum = 0.0f;
            for (int j = 0; j < cols_; ++j)
            {
                sum += powf((data_[i * cols_ + j] - mean(j, 0)), 2.0f);
            }
            result.data_[i] = sqrtf(sum / static_cast<float>(cols_));
        }
        return result;
    }
    throw std::invalid_argument("Invalid axis");
}

Matrix Matrix::elementwise_multiply(const Matrix &other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Matrix Matrix::broadcast_add(const Matrix &other, int axis) const
{
    // add row vector (1xcols) across all cols or column vector(1xrows) across all rows
    if (axis == 0 && other.cols() == cols() && other.rows() == 1)
    {
        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                result.data_[i * cols_ + j] = data_[i * cols_ + j] + other.data_[j];
            }
        }
        return result;
    }
    if (axis == 1 && other.rows() == rows() && other.cols() == 1)
    {
        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                result.data_[i * cols_ + j] = data_[i * cols_ + j] + other.data_[i];
            }
        }
        return result;
    }
    throw std::invalid_argument("Invalid axis or dimensions");
}

Matrix Matrix::max_over(int axis) const
{
    if (axis == 0)
    {
        Matrix result(1, cols_, InitMethod::ZERO);
        for (int j = 0; j < cols_; ++j)
        {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < rows_; ++i)
            {
                max_val = std::max(max_val, data_[i * cols_ + j]);
            }
            result.data_[j] = max_val;
        }
        return result;
    }
    if (axis == 1)
    {
        Matrix result(rows_, 1, InitMethod::ZERO);
        for (int i = 0; i < rows_; ++i)
        {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < cols_; ++j)
            {
                max_val = std::max(max_val, data_[i * cols_ + j]);
            }
            result.data_[i] = max_val;
        }
        return result;
    }
    throw std::invalid_argument("Invalid axis");
}

Matrix Matrix::broadcast_divide(const Matrix &other, int axis) const
{
    if (axis == 0 && other.cols() == cols() && other.rows() == 1)
    {
        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                result.data_[i * cols_ + j] = data_[i * cols_ + j] / other.data_[j];
            }
        }
        return result;
    }
    if (axis == 1 && other.rows() == rows() && other.cols() == 1)
    {
        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                // result(i, j) = data(i, j) / other(i, 0)
                result.data_[i * cols_ + j] = data_[i * cols_ + j] / other.data_[i];
            }
        }
        return result;
    }

    throw std::invalid_argument("Invalid axis or dimensions for broadcast_divide");
}
Matrix Matrix::argmax(int axis) const
{
    if (axis == 0)
    {
        Matrix result(1, cols_, Matrix::InitMethod::ZERO);
        for (int j = 0; j < cols_; ++j)
        {
            int max_i = 0;
            float max_val = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < rows_; ++i)
            {
                float val = get(i, j);
                if (val > max_val)
                {
                    max_val = val;
                    max_i = i;
                }
            }
            result.set(0, j, static_cast<float>(max_i));
        }
        return result;
    }
    if (axis == 1)
    {
        Matrix result(rows_, 1, Matrix::InitMethod::ZERO);
        for (int i = 0; i < rows_; ++i)
        {
            int max_j = 0;
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < cols_; ++j)
            {
                float val = get(i, j);
                if (val > max_val)
                {
                    max_val = val;
                    max_j = j;
                }
            }
            result.set(i, 0, static_cast<float>(max_j));
        }
        return result;
    }
    throw std::invalid_argument("Invalid axis for argmax: must be 0 or 1");
}