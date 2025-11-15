//
// Created by jakub-ciesko on 11/6/25.
//

#ifndef DATA_PREPARATOR_H
#define DATA_PREPARATOR_H

#include "../neural_network/matrix.h"
#include <random>
#include <string>

#include <vector>

class DataPreparator
{
  public:
    explicit DataPreparator(const std::string &data_root_path, int random_seed = 42,
                   int batch_size = 128);
    void load_data();
    void save_predictions(const Matrix &logits, const std::string &filename);
    void reset_epoch();
    void standardize_data();
    [[nodiscard]] bool has_next_batch() const;
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
    static Matrix load_vectors(const std::string &filename, int num_rows, int num_cols,
                               bool verbose);
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