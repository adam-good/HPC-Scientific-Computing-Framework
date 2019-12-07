#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>
#include "matrices.h"
#include "matrices-omp.h"
#include "matrices-cuda.h"

using namespace std;

void TimeTest(function<void()> test, string testName)
{
    double begin = omp_get_wtime();

    test();

    double end = omp_get_wtime();
    double elapsed_time = end - begin;

    cout << testName << "\n  Elapsed Time: " << elapsed_time << " Seconds" << endl;
}

void TestAdd()
{
    const unsigned int N = 3;
    array<int, N> shape = {50, 1000, 1000};

    auto test_serial = [=]() {
        HyperMatrix<N> A = HyperMatrix<N>::Zeros(shape);
        HyperMatrix<N> B = HyperMatrix<N>::Ones(shape);
        A + B;
    };

    auto test_omp = [=]() {
        HyperMatrix_OMP<N> A = HyperMatrix_OMP<N>::Zeros(shape);
        HyperMatrix_OMP<N> B = HyperMatrix_OMP<N>::Ones(shape);
        A + B;
    };

    auto test_cuda = [=]() {
        HyperMatrix_CUDA<N> A = HyperMatrix_CUDA<N>::Zeros(shape);
        HyperMatrix_CUDA<N> B = HyperMatrix_CUDA<N>::Ones(shape);
        A + B;
    };

    TimeTest(test_serial, "Serial Addition");
    TimeTest(test_omp, "OMP Addition");
    TimeTest(test_cuda, "CUDA Addition");
}

void TestScalar()
{
    const unsigned int N = 3;
    array<int,N> shape = {50,1000,1000};

    auto test_serial = [=]() {
        HyperMatrix<N> A = HyperMatrix<N>::Ones(shape);
        HyperMatrix<N>::ScalarMultiply(A, 8675309);
    };

    auto test_omp = [=]() {
        HyperMatrix_OMP<N> A = HyperMatrix_OMP<N>::Ones(shape);
        HyperMatrix_OMP<N>::ScalarMultiply(A, 8675309);
    };

    auto test_cuda = [=]() {
        HyperMatrix_CUDA<N> A = HyperMatrix_CUDA<N>::Ones(shape);
        HyperMatrix_CUDA<N>::ScalarMultiply(A, 8675309);
    };

    TimeTest(test_serial, "Serial Scalar Multiply");
    TimeTest(test_omp, "OMP Scalar Multiply");
    TimeTest(test_cuda, "CUDA Scalar Multiply");
}

void TestMultiplication()
{
    const unsigned int N = 3;
    array<int,N> shape = {4,500,500};

    auto test_serial = [=]() {
        HyperMatrix<N> A = HyperMatrix<N>::Ones(shape);
        HyperMatrix<N> B = HyperMatrix<N>::Ones(shape);
        A*B;
    };

    auto test_omp = [=]() {
        HyperMatrix_OMP<N> A = HyperMatrix_OMP<N>::Ones(shape);
        HyperMatrix_OMP<N> B = HyperMatrix_OMP<N>::Ones(shape);
        A*B;
    };

    auto test_cuda = [=]() {
        HyperMatrix_CUDA<N> A = HyperMatrix_CUDA<N>::Ones(shape);
        HyperMatrix_CUDA<N> B = HyperMatrix_CUDA<N>::Ones(shape);
        A*B;
    };

    TimeTest(test_serial, "Serial Multiply");
    TimeTest(test_omp, "OMP Multiply");
    TimeTest(test_cuda, "CUDA Multiply");
}


int main(void)
{
    // TestAdd();
    // TestScalar();
    TestMultiplication();

    // const unsigned int N = 2;
    // array<int, N> shapeA = {3,2};
    // array<int, N> shapeB = {2,3};
    // vector<double> valuesA = {1,2,3,4,5,6};
    // vector<double> valuesB = {7,8,9,10,11,12};

    // HyperMatrix_CUDA<N> A(shapeA, valuesA);
    // HyperMatrix_CUDA<N> B(shapeB, valuesB);

    // cout << A << endl;
    // cout << B << endl;
    // cout << A*B << endl;

    return 0;
}