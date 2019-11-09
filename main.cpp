#include <iostream>
#include <math.h>
#include "matrices.h"

using namespace std;


int main(void)
{
    const unsigned int N = 3;
    array<int, N> shape = {2,2,2};
    vector<double> values = {1,2,3,4,5,6,7,8};
    
    HyperMatrix<N> M = HyperMatrix<N>(shape, values);
    HyperMatrix<N> M_0 = HyperMatrix<N>::Zeros(shape);
    HyperMatrix<N> M_1 = HyperMatrix<N>::Ones(shape);

    cout << M.GetDims() << endl;
    cout << M << endl;
    cout << M_0 << endl;
    cout << M_1 << endl;
    cout << M.At({0,0,0}) << endl; // 1
    cout << M.At({1,0,0}) << endl; // 2
    cout << M.At({0,1,0}) << endl; // 3
    cout << M.At({1,1,0}) << endl; // 4
    cout << M.At({0,0,1}) << endl; // 5
    cout << M.At({1,0,1}) << endl; // 6
    cout << M.At({0,1,1}) << endl; // 7
    cout << M.At({1,1,1}) << endl; // 8

    return 0;
}