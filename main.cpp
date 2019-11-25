#include <iostream>
#include <math.h>
#include "matrices.h"

using namespace std;

int main(void)
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
    cout << B << endl;
    cout << endl;
    cout << C << endl;
    cout << endl;
    cout << I << endl;
    cout << endl;
    cout << D << endl;
    cout << endl;
    cout << M_0 << endl;

    return 0;
}