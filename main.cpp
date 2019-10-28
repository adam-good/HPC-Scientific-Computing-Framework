#include <iostream>
#include <math.h>
#include "matrices.h"

using namespace std;

int main(void)
{
    const unsigned int N = 3;
    array<int, N> shape = {2,2,3};
    vector<double> values = {1,2,3,4,5,6,7,8,9,10,11,12};
    
    HyperMatrix<N> M = HyperMatrix<N>(shape, values);
    HyperMatrix<N> M_0 = HyperMatrix<N>::Zeros(shape);

    cout << M.GetDims() << endl;
    cout << M << endl;

    return 0;
}