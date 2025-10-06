#include <iostream>
#include "src/matrix.h"

int main() {
    Matrix m1(3, 3);                  // defaults to ZERO
    Matrix m2(3, 3, Matrix::InitMethod::ONE);
    Matrix m3(3, 3, Matrix::InitMethod::NORMAL);
    m1.print();
    m2.print();
    m3.print();
    m3(0,0) = 10.0;
    float x = m3(1,1);
    x++;
    std::cout << x << std::endl;
    m3.print();
    Matrix m4 = m3.transpose();
    m4.print();
    m3.print();
    m4.transpose_inplace();
    m4.print();
    std::cout << std::endl;
    std::cout << "Vector" << std::endl;
    m4.print();
    std::cout << std::endl;
    Matrix m5(3, 1, Matrix::InitMethod::ZERO);
    Matrix m6(3, 1, Matrix::InitMethod::ONE);
    m5.print();
    std::cout << std::endl;
    ((m4 * m5 + m6).sum_over(1)).print();
    std::cout << std::endl;
    Matrix m7(3, 4, Matrix::InitMethod::ONE);
    m7.print();
    std::cout << std::endl;
    m7.sum_over(0).transpose().print();
    m7.rows();
    return 0;
}
