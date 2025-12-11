//
// Created by jakub-ciesko on 11/6/25.
//
#include "data_preparator.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../utils/train.h"


// TODO: add train-val-test split.


DataPreparator::DataPreparator(const std::string &data_root_path, int random_seed,
                               int batch_size, const std::string &file_prefix)
    : base_path(data_root_path), batch_size(batch_size), current_train_index(0),
      X_train(0, 0), X_test(0, 0), y_train_one_hot(0, 0), y_train(0, 0), y_test(0, 0),
      y_test_one_hot(0, 0), file_prefix(file_prefix)
{

    rng = std::mt19937(static_cast<unsigned int>(random_seed));
}

/**
 * loads CSV vectors and normalizes to [0,1]
 */

std::vector<std::vector<float>> load_csv(const std::string &filename,
                                         int num_rows = -1,
                                         bool verbose = false) {
    std::ifstream file(filename);
    if (!file.is_open()) throw std::runtime_error("Unable to open file");

    std::vector<std::vector<float>> rows;
    std::string line;
    int row_index = 0;
    int num_cols = -1;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string value;
        std::vector<float> row;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }

        if (num_cols == -1) num_cols = row.size(); // i dynamically get number of cols based on first row
        else if (row.size() != num_cols)
            throw std::runtime_error("Inconsistent number of columns in row " +
                                     std::to_string(row_index));

        rows.push_back(std::move(row));
        row_index++;

        if (num_rows != -1 && row_index >= num_rows) break;

        if (verbose && (row_index % 10000 == 0))
            std::cout << "\tRead " << row_index << " rows.\n";
    }
    file.close();
    if (verbose) std::cout << "Loaded " << rows.size() << " rows with "
                           << num_cols << " columns.\n";

    return rows;
}


Matrix DataPreparator::load_vectors(const std::string &filename, int num_rows,
                                    const bool verbose,
                                    const bool normalize_255_to_1)
{
    const std::vector<std::vector<float>> rows = load_csv(filename, num_rows, verbose);
    if (rows.empty()) throw std::runtime_error("Vector csv is empty at path: " + filename);
    const int num_cols = rows[0].size();
    Matrix data(rows.size(), num_cols);
    for (size_t i = 0; i < rows.size(); ++i)
        for (int j = 0; j < num_cols; ++j)
            data(i, j) = normalize_255_to_1? rows[i][j] / 255.0f : rows[i][j];
    return data;
}


/**
 * load CSV labels as integers or one-hot encoding
 */
Matrix DataPreparator::load_labels(const std::string &filename,
                                   int num_rows,
                                   const bool verbose)
{
    const std::vector<std::vector<float>> rows = load_csv(filename, num_rows, verbose);
    if (rows.empty()) throw std::runtime_error("Vector csv is empty at path: " + filename);
    const int num_cols = rows[0].size();
    Matrix data(rows.size(), num_cols);
    for (size_t i = 0; i < rows.size(); ++i)
        for (int j = 0; j < num_cols; ++j)
            data(i, j) = rows[i][j];
    return data;
}

/**
 * Load train and test data, initialize shuffled indices
 */
void DataPreparator::load_data()
{
    std::cout << "Loading data (prefix: " + file_prefix + ") from path: " + base_path << std::endl;

    const std::string train_vec_path    = file_prefix + "_train_vectors.csv";
    const std::string train_labels_path = file_prefix + "_train_labels.csv";
    const std::string test_vec_path     = file_prefix + "_test_vectors.csv";
    const std::string test_labels_path  = file_prefix + "_test_labels.csv";


    std::cout << "Loading train data vectors from " << train_vec_path << std::endl;
    X_train = load_vectors(base_path + train_vec_path,
                           -1,          // read all rows in the file
                          // 4,     // optionally parameterize for other datasets
                           true,        // verbose
                           false);
    std::cout << "Loading train data labels - classes from " << train_labels_path << std::endl;
    y_train = load_labels(base_path + train_labels_path,
                           -1,
                           true);
    std::cout << "Loading test data vectors from " << test_vec_path << std::endl;;
    X_test = load_vectors(base_path + test_vec_path,
                              -1,
                             // 4,
                              true,
                              false);
    std::cout << "Loading test data labels - classes from " << test_labels_path << std::endl;
    y_test = load_labels(base_path + test_labels_path,
                         -1,
                         true);

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

/**
 * Reset epoch: reshuffle train indices and reset pointer
 */
void DataPreparator::reset_epoch()
{
    current_train_index = 0;
    // reshuffling train indices to have different batches in this new epoch
    std::shuffle(train_indices_.begin(), train_indices_.end(), rng);
}

/**
 * returns next batch of data (X, y)
 */
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


/**
 * check if there are more batches in epoch
 */
bool DataPreparator::has_next_batch() const
{
    return current_train_index < static_cast<size_t>(X_train.rows());
}

/**
 * standardize train and test data using TRAIN statistics
 */
void DataPreparator::standardize_data()
{
    std::cout << "Standardizing train data" << std::endl;
    Matrix std = X_train.std_over(0);
    Matrix mean = X_train.mean_over(0);
    for (int i = 0; i < X_train.rows(); ++i)
    {
        for (int j = 0; j < X_train.cols(); ++j)
        {
            const float std_val = std.get(0, j);
            float z_score;
            if (std_val == 0.0f) {
                 z_score = 0.0f;
            }
            else {
                z_score = (X_train.get(i, j) - mean.get(0, j)) / std_val;
            }
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
            const float std_val = std.get(0, j);
            float z_score;
            if (std_val == 0.0f) {
                z_score = 0.0f;
            }
            else {
                z_score = (X_test.get(i, j) - mean.get(0, j)) / std_val;
            }
            X_test.set(i, j, z_score);
        }
    }
    std::cout << "Test data standardized" << std::endl
              << "All data standardized" << std::endl;
}

/**
 * Save predicted class labels to CSV file
 */
void DataPreparator::save_predictions(const Matrix &y_hat, const std::string &filename, const TaskDefinition &task) {
    const Matrix y_hat_write = task.task_type == Regression? y_hat : y_hat.argmax(1);
    std::cout << "Saving predictions to " << filename << std::endl;
    std::ofstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filename);
    }
    for (int i = 0; i < y_hat_write.rows(); ++i)
    {
        if (task.task_type == Regression) {
            float predicted_label = y_hat_write(i, 0);
            file << predicted_label << "\n";
        }
        else {
            int predicted_label = static_cast<int>(y_hat_write(i, 0));
            file << predicted_label << "\n";
        }
    }
    file.close();
}
