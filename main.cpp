#include <iostream>
#include <math.h>
#include "matrices.h"

using namespace std;

int main(void)
{
    const unsigned int N = 3;
    array<int, N> shape = {3,2,2};

    int size = 1;
    for (auto itr = shape.begin(); itr != shape.end(); itr++)
        size *= *itr;


    vector<double> values(size);
    for (int i = 1; i <= size; i++)
        values[i-1] = i;
    
    HyperMatrix<N> A = HyperMatrix<N>(shape, values);
    HyperMatrix<N> M_0 = HyperMatrix<N>::Zeros(shape);
    HyperMatrix<N> M_1 = HyperMatrix<N>::Ones(shape);

    // HyperMatrix<N> I = HyperMatrix<N>::Identity(shape);

    HyperMatrix<N> M = A;
    cout << M.GetDims() << endl;
    cout << M << endl;
    cout << M_0 << endl;
    cout << M_1 << endl;

    for (int k = 0; k < shape[2]; k++)
    for (int j = 0; j < shape[1]; j++)
    for (int i = 0; i < shape[0]; i++)
        {
            cout << "A[" << i << "," << j << "," << k << "] = " << M.At({i,j,k}) << endl;
        }

    return 0;
}