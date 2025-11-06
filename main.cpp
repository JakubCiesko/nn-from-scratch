#include <iostream>
#include "src/neural_network/matrix.h"
#include "src/data/data_preparator.h"

int main() {
    std::cout << "CSV READING" << std::endl;
    DataPreparator dp("../data/", 42);
    dp.load_data();
    dp.get_X_train().mean_over(0).print();
    dp.standardize_data();
    dp.get_X_train().mean_over(0).print();
    dp.get_X_train().std_over(0).print();
    return 0;
}
