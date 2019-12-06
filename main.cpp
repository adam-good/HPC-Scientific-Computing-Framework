#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>
#include "matrices.h"
#include "matrices-omp.h"

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
    array<int, N> shape = {100, 1000, 1000};

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

    TimeTest(test_serial, "Serial Addition");
    TimeTest(test_omp, "OMP Addition");
}

void TestScalar()
{
    const unsigned int N = 3;
    array<int,N> shape = {200,1000,1000};

    auto test_serial = [=]() {
        HyperMatrix<N> A = HyperMatrix<N>::Ones(shape);
        HyperMatrix<N>::ScalarMultiply(A, 8675309);
    };

    auto test_omp = [=]() {
        HyperMatrix_OMP<N> A = HyperMatrix_OMP<N>::Ones(shape);
        HyperMatrix_OMP<N>::ScalarMultiply(A, 8675309);
    };

    TimeTest(test_serial, "Serial Scalar Multiply");
    TimeTest(test_omp, "OMP Scalar Multiply");
}



int main(void)
{
    TestAdd();
    TestScalar();

    return 0;
}