#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>
#include "matrices.h"
#include "matrices-omp.h"
#include "matrices-cuda.h"

#define N 3

using namespace std;

void TimeTest(function<void()> test, string testName)
{
    double begin = omp_get_wtime();

    test();

    double end = omp_get_wtime();
    double elapsed_time = end - begin;

    cout << testName << "\n  Elapsed Time: " << elapsed_time << " Seconds" << endl;
}

void TestAdd(array<int, N> shape)
{
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

void TestScalar(array<int,N> shape)
{
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

void TestMultiplication(array<int,N> shape)
{
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

void TestSum()
{
    // const unsigned int N = 3;
    array<int,N> shape = {200,100,100};

    auto test_serial = [=]() {
        HyperMatrix<N> A = HyperMatrix<N>::Ones(shape);
        HyperMatrix<N>::Sum(A);
    };

    auto test_omp = [=]() {
        HyperMatrix_OMP<N> A = HyperMatrix_OMP<N>::Ones(shape);
        HyperMatrix_OMP<N>::Sum(A);
    };

    TimeTest(test_serial, "Serial Sum");
    TimeTest(test_omp, "OMP Sum");
}

double ApplySomething(double A)
{
    double B = A * 2 + 3;
    return B;
}
void TestApply()
{
    // const unsigned int N = 3;
    array<int,N> shape = {200,100,100};

    auto test_serial = [=]() {
        HyperMatrix<N> A = HyperMatrix<N>::Ones(shape);
        A.HyperMatrix<N>::Apply(ApplySomething);
    };

    auto test_omp = [=]() {
        HyperMatrix_OMP<N> A = HyperMatrix_OMP<N>::Ones(shape);
        A.HyperMatrix_OMP<N>::Apply(ApplySomething);
    };

    TimeTest(test_serial, "Serial Apply");
    TimeTest(test_omp, "OMP Apply");
}

int main(void)
{
    array<int,N> shape = {50,256,256};
    // TestAdd(shape);
    // TestScalar(shape);
    for (int i = 1; i <= 128; i += i)
    {
        shape = {i, 256, 256};
        TestMultiplication(shape);
    }
    // TestSum();
    // TestApply();

    return 0;
}
