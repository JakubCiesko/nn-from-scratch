//
// Created by jakub-ciesko on 11/6/25.
//

#ifndef DATA_PREPARATOR_H
#define DATA_PREPARATOR_H

#include "../neural_network/matrix.h"
#include <random>
#include <string>

#include <vector>

/**
* DataPreparator class loads, processes data.
* The class handles loading, standardizing, batching, train-test splitting, and saving of predictions.
* The file names are hardcoded and are relative to the provided data_root_path
*/
class DataPreparator
{
  public:
    /**
     * @param data_root_path Root path where dataset files are stored.
     * @param random_seed Seed for random number generator (default: 42).
     * @param batch_size Size of mini-batches for training (default: 128).
    */
    explicit DataPreparator(const std::string &data_root_path, int random_seed = 42,
                   int batch_size = 128);
    /**
    * load_data method loads csv data from defined data_root_path, splits it into train and test sets,
    * and initialized train_indices array which will hold order of train samples for training
    */
    void load_data();
    /**
     * Saves predictions to a file.
     * @param logits Predicted logits (logits! not class labels).
     * @param filename Output file path.
    */
    void save_predictions(const Matrix &logits, const std::string &filename);
    /** Resets current epoch and shuffles train indices. */
    void reset_epoch();
    /** Standardizes train and test data using train-data computed statistics. */
    void standardize_data();
    /** Returns true if more batches are available. Used for looping in training. */
    [[nodiscard]] bool has_next_batch() const;
    /** Returns next batch of train data. */
    std::pair<Matrix, Matrix> get_batch();
    [[nodiscard]] const Matrix &get_X_train() const
    {
        return X_train;
    }
    [[nodiscard]] const Matrix &get_X_test() const
    {
        return X_test;
    }
    [[nodiscard]] const Matrix &get_y_train() const
    {
        return y_train;
    }
    [[nodiscard]] const Matrix &get_y_test() const
    {
        return y_test;
    }

  private:
    /** Loads vectors from a single csv file.
    * @param filename csv file name
    * @param num_rows # of rows to load
    * @param num_cols # of cols in csv file
    * @param verbose verbosity flag
    * @return Matrix of data
    */
    static Matrix load_vectors(const std::string &filename, int num_rows, int num_cols,
                               bool verbose);
    /** Loads labels from csv file.
    * @param filename Path to csv file.
    * @param num_rows Number of rows to load.
    * @param as_one_hot If true, returns one-hot encoded labels.
    * @param num_classes Number of classes (used for one-hot).
    * @return Matrix of labels.
    */
    static Matrix load_labels(const std::string &filename, int num_rows,
                              bool as_one_hot, int num_classes = 10);
    std::string base_path;
    std::mt19937 rng;
    Matrix X_train;
    Matrix X_test;
    Matrix y_train_one_hot;
    Matrix y_train;
    Matrix y_test_one_hot;
    Matrix y_test;
    std::vector<int> train_indices_;
    size_t current_train_index;
    int batch_size;
};

#endif // DATA_PREPARATOR_H