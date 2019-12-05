#include <iostream>
#include <cmath>
#include <ctime>
#include <omp.h>
#include "matrices.h"
#include "matrices-omp.h"

using namespace std;

void TestAdd()
{
    const unsigned int N = 3;
    array<int, N> shape = {3, 300, 300};

    HyperMatrix<N> A = HyperMatrix<N>::Zeros(shape);
    HyperMatrix<N> B = HyperMatrix<N>::Ones(shape);

    A + B;
}

void TestAdd_OMP()
{
    const unsigned int N = 3;
    array<int, N> shape = {3, 300, 300};

    HyperMatrix_OMP<N> A = HyperMatrix_OMP<N>::Zeros(shape);
    HyperMatrix_OMP<N> B = HyperMatrix_OMP<N>::Ones(shape);

    A + B;
}

void TimeTest(function<void()> test, string testName)
{
    double begin = omp_get_wtime();

    test();

    double end = omp_get_wtime();
    double elapsed_time = end - begin;

    cout << testName << "\n  Elapsed Time: " << elapsed_time << " Seconds" << endl;
}

int main(void)
{
    TimeTest(TestAdd, "Serial Addition");
    TimeTest(TestAdd_OMP, "OMP Addition");
    return 0;
}