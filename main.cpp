#include <iostream>
#include <cmath>
#include <ctime>
#include "matrices.h"

using namespace std;

void TestApply()
{
    const unsigned int N = 2;

    array<int, N> shape_A = {5,5};

    int size_A = 1;
    for (auto itr = shape_A.begin(); itr != shape_A.end(); itr++) size_A *= *itr;

    vector<double> values_A(size_A);
    for (int i = 0; i < size_A; i++) values_A[i] = i;

    HyperMatrix<N> A(shape_A, values_A);

    auto sigmoid = [](double x) { return 1.0 / (1.0 + exp(-x) ); };

    cout << A << endl;
    cout << A.Apply([](double x){ return sin(x);} ) << endl;
}

void TestArithmatic()
{
    const unsigned int N = 2;
    array<int, N> shape_A = {3,2};
    array<int, N> shape_B = {2,3};

    int size = 1;
    for (auto itr = shape_A.begin(); itr != shape_A.end(); itr++)
        size *= *itr;


    vector<double> values_A(size);
    vector<double> values_B(size);
    for (int i = 0; i < size; i++)
    {
        values_A[i] = i;
        values_B[i] = size-i;    
    }

    HyperMatrix<N> A(shape_A, values_A);
    HyperMatrix<N> B(shape_B, values_B);
    HyperMatrix<N> C = A*B;
    HyperMatrix<N> I = HyperMatrix<N>::Identity(C.getShape());
    HyperMatrix<N> D = C + I;
    HyperMatrix<N> M_0 = I - I;


    cout << A << endl;
    cout << endl;
    cout << B << endl;
    cout << endl;
    cout << C << endl;
    cout << endl;
    cout << I << endl;
    cout << endl;
    cout << D << endl;
    cout << endl;
    cout << M_0 << endl;
}

void TestScale()
{
    const unsigned int N = 2;
    int n = 4096;
    int m = 2160;
    int z = 3;
    array<int, N> shape_A = {n,m};
    array<int, N> shape_B = {m,n};

    HyperMatrix<N> A = HyperMatrix<N>::Zeros(shape_A);
    HyperMatrix<N> B = HyperMatrix<N>::Ones(shape_B);

    A*B;
}

void TimeTest(function<void()> test, string testName)
{
    clock_t begin = clock();

    test();

    clock_t end = clock();
    double elapsed_time = double(end - begin) / CLOCKS_PER_SEC;

    cout << testName << "\n  Elapsed Time: " << elapsed_time << " Seconds" << endl;
}

int main(void)
{
    // TimeTest(TestArithmatic, "Arithmatic");
    // TimeTest(TestApply, "Apply");
    TimeTest(TestScale, "Scale");
    return 0;
}