//
// Created by jakub-ciesko on 11/6/25.
//

#ifndef DATA_PREPARATOR_H
#define DATA_PREPARATOR_H

#include "../neural_network/matrix.h"
#include<string>
#include<vector>
#include<random>
#include<utility>

class DataPreparator {
public:
    DataPreparator(const std::string& data_root_path, int random_seed = 42);
    void load_data();
    void reset_epoch();
    void standardize_data();
    // std::pair<Matrix, Matrix> get_next_train_batch(int batch_size);
    [[nodiscard]] const Matrix& get_X_train() const { return X_train; }
    [[nodiscard]] const Matrix& get_X_test() const { return X_test; }
    [[nodiscard]] const Matrix& get_y_train() const { return y_train; }
    [[nodiscard]] const Matrix& get_y_test() const { return y_test; }

private:
    static Matrix load_vectors(const std::string& filename, int num_rows, int num_cols, bool verbose);
    static Matrix load_labels(const std::string& filename, int num_rows, bool as_one_hot, int num_classes = 10);
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
};

#endif //DATA_PREPARATOR_H