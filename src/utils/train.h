//
// Created by jakub-ciesko on 12/5/25.
//

#pragma once
#include <string>
#include "../data/data_preparator.h"
#include "../neural_network/network.h"
#include "../neural_network/optimizer.h"
#include "../neural_network/matrix.h"

// helpers
std::string current_time();
float compute_accuracy(const Matrix &logits, const Matrix &y_true);

// training / prediction
void prepare_data(DataPreparator &data_preparator, bool standardize_data);

struct TrainingParams {
    int epochs;
    int batch_size;
    float adam_lr;
    float adam_beta1;
    float adam_beta2;
    float weight_decay;
    float epsilon;
    bool standardize_data;
    bool use_dropout;
    float dropout_p;
};

enum TaskType {Classification, Regression};
struct TaskDefinition {
    TaskType task_type;
    std::string task_name;
    int final_layer_dim;
    bool normalize_255_to_1;
};

void train(TrainingParams &training_params, Network &network,
           Optimizer &optimizer, DataPreparator &data_preparator, const TaskDefinition &task);

// training using preallocated tensors -- this does not work correctly currently
void train_prealloc(TrainingParams &training_params, Network &network,
                    Optimizer &optimizer, DataPreparator &data_preparator);

void predict(Network &network, DataPreparator &data_preparator,
             bool is_test, const std::string &filename, const TaskDefinition &task);