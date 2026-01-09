//
// Created by jakub-ciesko on 9/23/25.
//
#include "matrix.h"
#include <iostream>
#include <random>
#include <stdexcept>

// init code
Matrix::Matrix(int rows, int cols, InitMethod method, std::mt19937* gen)
    : rows_(rows), cols_(cols), data_(rows * cols)
{
    initialize(method, gen);
}
/**
 * Initializes matrix values to predefined values based on provided (enum) method
 */
void Matrix::initialize(InitMethod method, std::mt19937* gen)
{
    switch (method)
    {
        case InitMethod::ZERO:
            std::fill(data_.begin(), data_.end(), 0.0f);
            break;

        case InitMethod::ONE:
            // I used this mainly for experiments whether i coded matrix algebra right
            std::fill(data_.begin(), data_.end(), 1.0f);
            break;

        case InitMethod::NORMAL:
        {
            if (gen == nullptr) {
                throw std::invalid_argument("Generator cannot be null for NORMAL init");
            }
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto &x : data_)
                x = dist(*gen);
            break;
        }

        case InitMethod::KAIMING:
        {
            if (gen == nullptr) {
                throw std::invalid_argument("Generator cannot be null for NORMAL init");
            }
            float stddev = sqrtf(2.0f / static_cast<float>(rows_)); // n_features_in = cols_
            // mathematically equivalent to just ... dist(0.0f, stddev); but it keeps the code style as in NORMAL
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (auto &x : data_)
                x = dist(*gen) * stddev;
            break;
        }
    }
}

/**
* checks whether dims of two matrices are the same. used for elementwise operators such as +, -, elementwise *, ...
*/
void Matrix::check_dims_match_(const Matrix &other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match. Got " + shape_str_() + " and " + other.shape_str_());
    }
}
/**
* checks matrix dimension for matrix multiplication
*/
void Matrix::check_dims_matmul_(const Matrix &other) const {
    if (cols_ != other.rows_)
        throw std::invalid_argument("Invalid matrix dimensions for matmul. Got " + shape_str_() + " and " + other.shape_str_());
}
/**
* checks whether provided indices are inside matrix limits
*/
void Matrix::check_element_indices_(int row, int col) const {
    if (row < 0 || row >= rows_ || col < 0 || col >= cols_)
        throw std::out_of_range("Index out of range.");
}

/**
* helper method for displaying messages about shape mismatches
*/
std::string Matrix::shape_str_() const {
    return "(" + std::to_string(rows_) +"," + std::to_string(cols_) + ")";
}

/**
* checks whether row, col inside matrix bounds and sets value at this place
*/
void Matrix::set(int row, int col, float value)
{
    check_element_indices_(row, col);
    data_[row * cols_ + col] = value;
}

/**
* checks whether row, col inside matrix bounds and gets value from this place
*/
float Matrix::get(int row, int col) const
{
    check_element_indices_(row, col);
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

/**
* () operator for getter / setter
*/
float &Matrix::operator()(int row, int col)
{
    check_element_indices_(row, col);
    return data_[row * cols_ + col];
}
float Matrix::operator()(int row, int col) const
{
    check_element_indices_(row, col);
    return data_[row * cols_ + col];
}

/**
 *helper function for viewing matrices. Use for debugging purposes, do not use for big matrices.
 */
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


// basic matrix algebra

/**
 * Matrix + Matrix
 */
Matrix Matrix::operator+(const Matrix &other) const
{
    check_dims_match_(other);
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

/**
 * Matrix + float
 */
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
    check_dims_match_(other);
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

/**
 * Matrix multiplication operator. Uses parallelized computing with omp parallel
 * Could be enhanced for speed with tiling or other more effective algorithms.
 */
Matrix Matrix::operator*(const Matrix &other) const
{
    check_dims_matmul_(other);
    Matrix result(rows_, other.cols_);
// Cij = dot(Ai*, B*,j) = sum(AiK*bKj)
// or more effectively switching the order or ijk to ikj to have AiK preloaded
// https://stackoverflow.com/questions/20467117/for-matrix-operation-why-is-ikj-faster-than-ijk
#pragma omp parallel for collapse(2) default(none) shared(result, other, rows_, cols_)

    for (int i = 0; i < rows_; ++i)
    {
        for (int k = 0; k < cols_; ++k)
        {
            float a = data_[i * cols_ + k];
            for (int j = 0; j < other.cols_; ++j)
            {
                result.data_[i * other.cols_ + j] +=
                    a * other.data_[k * other.cols_ + j];
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

/**
 * changes matrix M to M^T inplace (the old structure will be lost), but remember M = (M^T)^T
 */
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
    const int temp = rows_;
    rows_ = cols_;
    cols_ = temp;
}

/**
 * returns transposed copy of Matrix M
 */
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

/**
 * Applies float-valued function to elements of matrix (can be used for relu for example). Returns copy of matrix.
 */
Matrix Matrix::apply(const std::function<float(float)>& function) const
{
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = function(data_[i]);
    }
    return result;
}

/**
 * Applies float-valued function to elements of matrix. Changes the original matrix values.
 */
void Matrix::apply_inplace(const std::function<float(float)>& function)
{
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        data_[i] = function(data_[i]);
    }
}

/**
 * Returns copy of flattened 1xCols or 1xRows matrix with sums over rows or columns.
 * Tries to mimic numpy array method.
 * @param axis 0 is sum over rows (colsx1), 1 is sum over cols (rowsx1)
 * @return new matrix with reduced dimension
 */
Matrix Matrix::sum_over(int axis) const
{
    if (axis == 0)
    {
        Matrix result(1, cols_, InitMethod::ZERO);
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                result.data_[j] += data_[i * cols_ + j];
            }
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

/**
 * Returns new matrix with reduced dimensions. axis 0 is mean over rows (colsx1), 1 is mean over cols (rowsx1)
 */
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

/**
 * Returns new matrix with reduced dimensions. axis 0 is std (standard deviation) over rows (colsx1), 1 is std over cols (rowsx1)
 */
Matrix Matrix::std_over(int axis) const
{
    Matrix mean = mean_over(axis);
    if (axis == 0)
    {
        // sum over rows, matrix is 1 * columns
        Matrix std(1, cols_, InitMethod::ZERO);
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {

                float diff = data_[i * cols_ + j] - mean.get(0, j);
                std.data_[j] += powf(diff, 2.0f);
            }
        }
        const float inverse_N = 1.0f / static_cast<float>(rows_);
        for (int j = 0; j < cols_; ++j)
        {
            std.data_[j] = sqrtf(std.data_[j] * inverse_N);
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
            float mean_val = mean(i, 0);
            for (int j = 0; j < cols_; ++j)
            {
                sum += powf((data_[i * cols_ + j] - mean_val), 2.0f); //  j -> i
            }
            result.data_[i] = sqrtf(sum / static_cast<float>(cols_));
        }
        return result;
    }
    throw std::invalid_argument("Invalid axis");
}

/**
 * Returns new matrix Cij = Aij*Bij -- elementwise multiplies with other matrix.
 * Matrices must have the same shape!
 */
Matrix Matrix::elementwise_multiply(const Matrix &other) const
{
    check_dims_match_(other);
    Matrix result(rows_, cols_);
    for (int i = 0; i < rows_ * cols_; ++i)
    {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}


float Matrix::max() const {
    float max = data_[0];
    for (const float& el : data_) {
        if (el > max)
            max = el;
    }
    return max;
}


float Matrix::mean() const {
    float sum = 0.0f;
    for (const float& el : data_)
        sum += el;
    const float N = static_cast<float>(rows_*cols_);
    return sum / N;
}

float Matrix::std() const {
    const float mean_value = mean();
    float sum = 0.0f;
    for (const float& el : data_)
        sum += powf(el - mean_value, 2.0f);
    const float N = static_cast<float>(rows_*cols_);
    return sqrtf(sum / N);
}

float Matrix::sum() const {
    float sum = 0.0f;
    for (const float& el : data_)
        sum += el;
    return sum;
}

/**
 * Adds the values of `other` to this matrix along the specified axis, using broadcasting rules similar to NumPy:
 * - If `axis == 0` and `other` is a 1×cols vector, it is added to each row.
 * - If `axis == 1` and `other` is a rows×1 vector, it is added to each column.
 */
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

/**
 * Returns new matrix with reduced dimensions. axis 0 is max (standard deviation) over rows (colsx1), 1 is max over cols (rowsx1)
 */
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

/**
 * Divides elements of matrix by provided Matrix (which is a vector) along the specified axis.
 * - If `axis == 0` and `other` is a 1×cols vector, each row of this matrix is divided elementwise by `other`.
 * - If `axis == 1` and `other` is a rows×1 vector, each column of this matrix is divided elementwise by `other`.
 * Used in softmax calculation.
*/
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

/**
 * Returns new matrix with reduced dimensions. axis 0 is argmax (standard deviation) over rows (colsx1), 1 is argmax over cols (rowsx1)
 */
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

/**
 * Computes A*B + C in fused manner to reduce computational load of first D = A*B and then D + C.
 * @param B Matrix to be multiplied with
 * @param C Matrix to be added
 * @return Matrix with value of A*B + C
 */
Matrix Matrix::matmul_broadcast_add(const Matrix &B, const Matrix &C) const
{
    // A=Batch*m B=m*r C = Batch*1 -> result = batch*r
    if (cols_ != B.rows())
        throw std::invalid_argument("Invalid matrix dimensions");
    Matrix result(rows_, B.cols());

    #pragma omp parallel for
    for (int i = 0; i < rows_; ++i)
        for (int j = 0; j < B.cols_; ++j)
            result.data_[i * B.cols_ + j] = C.data_[j];


    #pragma omp parallel for
    for (int i = 0; i < rows_; ++i)
        for (int k = 0; k < cols_; ++k)
        {
            float a = data_[i * cols_ + k];
            for (int j = 0; j < B.cols_; ++j)
                result.data_[i * B.cols_ + j] += a * B.data_[k * B.cols_ + j];
        }
    return result;
}

/*
 * these methods bypass matrix creation on each method pass, they use preallocated resources
 */

/**
 * Matrix multiplication with result saved into preallocated matrix.
 */
void Matrix::matmul(const Matrix &other, Matrix &result) const {

        check_dims_matmul_(other);
#pragma omp parallel for collapse(2) default(none) shared(result, other, rows_, cols_)

         for (int i = 0; i < rows_; ++i)
        {
             for (int j = 0; j < other.cols_; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < cols_; ++k)
                {
                    sum += data_[i * cols_ + k] * other.data_[k * other.cols_ + j];
                }
                result.data_[i * other.cols_ + j] = sum;
            }
        }
}

/**
 * Computes A*B + C in fused manner to reduce computational load of first D = A*B and then D + C.
 * Uses preallocated resources
 * @param B Matrix to be multiplied with
 * @param C Matrix to be added
 * @param result Matrix in which result is saved
 * @return Matrix with value of A*B + C
 */
void Matrix::matmul_broadcast_add_prealloc(const Matrix &B, const Matrix &C, Matrix &result) const {

    if (cols_ != B.rows())
        throw std::invalid_argument("Invalid matrix dimensions");

#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows_; ++i)
    {
        for (int j = 0; j < B.cols_; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < cols_; ++k)
            {
                sum += data_[i * cols_ + k] * B.data_[k * B.cols_ + j];
            }
            result.data_[i * B.cols_ + j] = sum + C.data_[j];
        }
    }
}

/**
 * Fastest way to fill matrix with one float value. Faster than Matrix.apply_inplace([](float x){return val;})
 */
void Matrix::fill(const float value) {
    std::fill(data_.begin(), data_.end(), value);
}
