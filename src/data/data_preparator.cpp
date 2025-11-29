//
// Created by jakub-ciesko on 11/6/25.
//
#include "data_preparator.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

DataPreparator::DataPreparator(const std::string &data_root_path, int random_seed,
                               int batch_size)
    : base_path(data_root_path), batch_size(batch_size), current_train_index(0),
      X_train(0, 0), X_test(0, 0), y_train_one_hot(0, 0), y_train(0, 0), y_test(0, 0),
      y_test_one_hot(0, 0)
{

    rng = std::mt19937(static_cast<unsigned int>(random_seed));
}

Matrix DataPreparator::load_vectors(const std::string &filename, int num_rows,
                                    int num_cols, bool verbose)
{
    // will use this for loading data into it
    Matrix data(num_rows, num_cols, Matrix::InitMethod::ZERO);
    std::ifstream file(filename.c_str());
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file");
    }

    std::string line;
    std::string value;
    // line by line parsing of csv values
    for (int i = 0; i < num_rows; ++i)
    {
        if (!std::getline(file, line))
        {
            throw std::runtime_error("File error: Not enough rows. Expected " +
                                     std::to_string(num_rows));
        }
        std::stringstream ss(line);

        for (int j = 0; j < num_cols; ++j)
        {
            if (!std::getline(ss, value, ','))
            {
                throw std::runtime_error("File error: Not enough columns in row " +
                                         std::to_string(i));
            }
            data(i, j) = std::stof(value) / 255.0f; // normalization
        }
        if (verbose && (i + 1) % 10000 == 0)
        {
            std::cout << "  ... read " << (i + 1) << " vector lines." << std::endl;
        }
    }
    file.close();
    return data;
}

Matrix DataPreparator::load_labels(const std::string &filename, int num_rows,
                                   bool as_one_hot, int num_classes)
{
    int num_cols = as_one_hot ? num_classes : 1;
    Matrix data(num_rows, num_cols, Matrix::InitMethod::ZERO);
    std::ifstream file(filename.c_str());
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file");
    }
    std::string line;
    for (int i = 0; i < num_rows; ++i)
    {
        if (!std::getline(file, line))
        {
            throw std::runtime_error("File error: Not enough rows. Expected " +
                                     std::to_string(num_rows) +
                                     " Got: " + std::to_string(i));
        }
        std::stringstream ss(line);
        int value = std::stoi(line);
        if (as_one_hot)
        {
            data(i, value) = 1.0f;
        }
        else
        {
            data(i, 0) = static_cast<float>(value);
        }
    }
    file.close();
    return data;
}

void DataPreparator::load_data()
{
    std::cout << "Loading data from path: " + base_path << std::endl;
    std::cout << "Loading train data vectors (60 000 vecs)" << std::endl;
    X_train = load_vectors(base_path + "fashion_mnist_train_vectors.csv", 60000,
                           28 * 28, false);
    std::cout << "Loading train data labels (60 000 labels) - 10 classes" << std::endl;
    y_train = load_labels(base_path + "fashion_mnist_train_labels.csv", 60000, false, 10);
    std::cout << "Loading test data vectors (10 000 vecs)" << std::endl;
    X_test = load_vectors(base_path + "fashion_mnist_test_vectors.csv", 10000, 28 * 28,
                          false);
    std::cout << "Loading test data labels (10 000 vecs) - 10 classes" << std::endl;
    y_test = load_labels(base_path + "fashion_mnist_test_labels.csv", 10000, false, 10);

    // the random order of train samples is achieved through using randomly shuffled train indices
    // shuffling occrus in reset_epoch method
    train_indices_.resize(X_train.rows());

    for (int i = 0; i < X_train.rows(); ++i)
    {
        train_indices_[i] = i;
    }
    reset_epoch();
    std::cout << "All data loaded" << std::endl;
}

void DataPreparator::reset_epoch()
{
    current_train_index = 0;
    // reshuffling train indices to have different batches in this new epoch
    std::shuffle(train_indices_.begin(), train_indices_.end(), rng);
}

std::pair<Matrix, Matrix> DataPreparator::get_batch()
{
    size_t from = current_train_index;
    // set batch size or smaller when not enough data
    size_t to = std::min(from + static_cast<size_t>(batch_size),
                         static_cast<size_t>(X_train.rows()));
    size_t actual_batch_size = to - from;
    // what if we go out of bounds
    Matrix X_batch(actual_batch_size, X_train.cols());
    Matrix y_batch(actual_batch_size, y_train.cols());
    for (size_t i = 0; i < actual_batch_size; ++i)
    {
        int shuffled_index = train_indices_[from + i];
        for (int j = 0; j < X_train.cols(); ++j)
        {
            X_batch(i, j) = X_train.get(shuffled_index, j);
        }

        for (int j = 0; j < y_train.cols(); ++j)
        {
            y_batch(i, j) = y_train.get(shuffled_index, j);
        }
    }
    // move to new train index
    current_train_index = to;
    return {X_batch, y_batch};
}

bool DataPreparator::has_next_batch() const
{
    return current_train_index < static_cast<size_t>(X_train.rows());
}

void DataPreparator::standardize_data()
{
    std::cout << "Standardizing train data" << std::endl;
    Matrix std = X_train.std_over(0);
    Matrix mean = X_train.mean_over(0);
    for (int i = 0; i < X_train.rows(); ++i)
    {
        for (int j = 0; j < X_train.cols(); ++j)
        {
            float z_score = (X_train.get(i, j) - mean.get(0, j)) / std.get(0, j);
            X_train.set(i, j, z_score);
        }
    }
    std::cout << "Train data standardized" << std::endl;
    // careful here, test data must be standardized with train-data computed statistics
    std::cout << "Standardizing test data" << std::endl;
    for (int i = 0; i < X_test.rows(); ++i)
    {
        for (int j = 0; j < X_test.cols(); ++j)
        {
            float z_score = (X_test.get(i, j) - mean.get(0, j)) / std.get(0, j);
            X_test.set(i, j, z_score);
        }
    }
    std::cout << "Test data standardized" << std::endl
              << "All data standardized" << std::endl;
}

void DataPreparator::save_predictions(const Matrix &logits, const std::string &filename) {
    // inputs are logits, outputs are class labels got as output from argmax over columns.
    Matrix y_hat = logits.argmax(1);
    std::cout << "Saving predictions to " << filename << std::endl;
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }
    for (int i = 0; i < y_hat.rows(); ++i)
    {
        int predicted_label = static_cast<int>(y_hat(i, 0));
        file << predicted_label << "\n";
    }
    file.close();
}
