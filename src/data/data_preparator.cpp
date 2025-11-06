//
// Created by jakub-ciesko on 11/6/25.
//
#include "data_preparator.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

DataPreparator::DataPreparator(const std::string &data_root_path, int random_seed)
    : base_path(data_root_path), current_train_index(0), X_train(0, 0), X_test(0, 0),
      y_train_one_hot(0, 0), y_train(0, 0), y_test(0, 0), y_test_one_hot(0, 0)
{

    if (random_seed == -1)
    {
        std::random_device rd;
        rng = std::mt19937(rd());
    }
    else
    {
        rng = std::mt19937(static_cast<unsigned int>(random_seed));
    }
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
    std::cout << "Loading train data vectors (60 000 vecs)" << std::endl;
    X_train = load_vectors(base_path + "fashion_mnist_train_vectors.csv", 60000,
                           28 * 28, false);
    std::cout << "Loading train data labels (60 000 labels) - 10 classes" << std::endl;
    y_train =
        load_labels(base_path + "fashion_mnist_train_labels.csv", 60000, false, 10);
    std::cout << "Loading test data vectors (10 000 vecs)" << std::endl;
    X_test = load_vectors(base_path + "fashion_mnist_test_vectors.csv", 10000, 28 * 28,
                          false);
    std::cout << "Loading test data labels (10 000 vecs) - 10 classes" << std::endl;
    y_test = load_labels(base_path + "fashion_mnist_test_labels.csv", 10000, false, 10);
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
    std::shuffle(train_indices_.begin(), train_indices_.end(), rng);
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
    // oopaaa, mind that data leakage :D
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
